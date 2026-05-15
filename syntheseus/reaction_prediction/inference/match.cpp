#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rdkit/GraphMol/ROMol.h>
#include <rdkit/GraphMol/SmilesParse/SmilesParse.h>
#include <rdkit/GraphMol/SmilesParse/SmilesWrite.h>
#include <rdkit/GraphMol/RingInfo.h>
#include <rdkit/GraphMol/Atom.h>
#include <rdchiral/rdchiral.hpp>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <cmath>
#include <climits>
#include <tuple>
#include <memory>
#include <fstream>
#include <iostream>

namespace py = pybind11;

// ============================================================================
// Configuration
// ============================================================================

struct MatchConfig {
    float w_cd = 0.1f;
    float w_as = 0.2f;
    float w_rd = 0.5f;
    float w_md = 0.0f;
    float w_private = 0.0f;
    float private_bonus = 10.0f;
    std::unordered_set<std::string> private_templates;

    bool is_private(const std::string& tpl) const {
        return private_templates.count(tpl) > 0;
    }
};

// ============================================================================
// Template Library — pre-parsed templates + inventory, constructed once at init
// ============================================================================

class TemplateLibrary {
public:
    explicit TemplateLibrary(const std::vector<std::string>& templates) {
        reactions.reserve(templates.size());
        raw_templates.reserve(templates.size());
        for (const auto& t : templates) {
            reactions.emplace_back(std::make_unique<rdchiral::Reaction>(t));
            raw_templates.push_back(t);
        }
    }

    int size() const { return static_cast<int>(reactions.size()); }

    rdchiral::Reaction& get(int idx) { return *reactions[idx]; }
    const std::string& raw(int idx) const { return raw_templates[idx]; }

    // Load inventory directly from file — bypasses pybind11 transfer overhead
    void set_inventory_file(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Warning: cannot open inventory file: " << path << std::endl;
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            size_t end = line.find_last_not_of(" \t\r\n");
            std::string smi = line.substr(start, end - start + 1);
            if (!smi.empty()) {
                in_stock.insert(std::move(smi));
            }
        }
        std::cerr << "Loaded " << in_stock.size() << " inventory molecules from " << path << std::endl;
    }

    // Load private templates from JSON file — bypasses pybind11 transfer
    void set_private_templates_file(const std::string& path) {
        // Simple JSON array parser: skip brackets/quotes/commas, extract strings
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Warning: cannot open private templates file: " << path << std::endl;
            return;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        // Parse JSON array of strings
        size_t pos = 0;
        while ((pos = content.find('"', pos)) != std::string::npos) {
            size_t end = content.find('"', pos + 1);
            if (end == std::string::npos) break;
            std::string tpl = content.substr(pos + 1, end - pos - 1);
            // Skip empty strings and non-template entries (like bare "[" tokens)
            if (!tpl.empty() && tpl.find('[') != std::string::npos) {
                private_templates.insert(std::move(tpl));
            }
            pos = end + 1;
        }
        std::cerr << "Loaded " << private_templates.size() << " private templates from " << path << std::endl;
    }

    bool is_private(const std::string& tpl) const {
        return private_templates.count(tpl) > 0;
    }

    const std::unordered_set<std::string>& get_inventory() const {
        return in_stock;
    }

private:
    std::vector<std::unique_ptr<rdchiral::Reaction>> reactions;
    std::vector<std::string> raw_templates;
    std::unordered_set<std::string> in_stock;
    std::unordered_set<std::string> private_templates;
};

// ============================================================================
// Utility functions (pure C++)
// ============================================================================

inline int get_mapped_atom_count(const RDKit::ROMol& mol) {
    int count = 0;
    for (const auto atom : mol.atoms()) {
        int map = atom->getAtomMapNum();
        if (map > 0 && map < 900) count++;
    }
    return count;
}

inline std::vector<std::string> split_reactants(const std::string& s) {
    std::vector<std::string> result;
    std::string::size_type start = 0, end;
    while ((end = s.find('.', start)) != std::string::npos) {
        if (end > start) result.emplace_back(s.substr(start, end - start));
        start = end + 1;
    }
    if (start < s.size()) result.emplace_back(s.substr(start));
    return result;
}

inline bool has_element(const std::string& smiles, const char* elem) {
    return smiles.find(elem) != std::string::npos;
}

inline std::string canonical_without_map(const RDKit::ROMol& mol) {
    RDKit::ROMol mol_copy(mol);
    for (auto atom : mol_copy.atoms()) {
        atom->setAtomMapNum(0);
    }
    std::string smi = RDKit::MolToSmiles(mol_copy);
    std::unique_ptr<RDKit::ROMol> mol2(RDKit::SmilesToMol(smi));
    if (!mol2) return "";
    return RDKit::MolToSmiles(*mol2);
}

inline std::string canonical_from_smiles(const std::string& smi) {
    std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smi));
    if (!mol) return "";
    for (auto atom : mol->atoms()) {
        atom->setAtomMapNum(0);
    }
    std::string result = RDKit::MolToSmiles(*mol);
    std::unique_ptr<RDKit::ROMol> mol2(RDKit::SmilesToMol(result));
    if (!mol2) return "";
    return RDKit::MolToSmiles(*mol2);
}

// ============================================================================
// Scoring functions (pure C++, no Python calls)
// ============================================================================

float calc_cdscore(const RDKit::ROMol& p_mol,
                   const std::vector<std::string>& reactants,
                   const std::vector<RDKit::ROMol*>& reactant_mols) {
    int p_count = static_cast<int>(p_mol.getNumAtoms());
    int n = static_cast<int>(reactants.size());
    if (n <= 1) return 0.0f;

    std::vector<int> r_counts(n);
    for (int i = 0; i < n; i++) {
        r_counts[i] = get_mapped_atom_count(*reactant_mols[i]);
    }

    int main_idx = 0;
    for (int i = 1; i < n; i++) {
        if (r_counts[i] > r_counts[main_idx]) main_idx = i;
    }

    if (static_cast<int>(reactant_mols[main_idx]->getNumAtoms()) >= p_count) {
        return 0.0f;
    }

    double avg = static_cast<double>(p_count) / n;
    double mae = 0.0;
    for (int i = 0; i < n; i++) {
        mae += std::abs(avg - r_counts[i]);
    }
    mae /= n;

    return static_cast<float>(p_count / (1.0 + mae));
}

float calc_asscore(
    const RDKit::ROMol& p_mol,
    const std::vector<RDKit::ROMol*>& reactant_mols,
    const std::vector<std::string>& reactant_canonical_smiles,
    const std::unordered_set<std::string>& in_stock
) {
    int p_count = static_cast<int>(p_mol.getNumAtoms());
    int n = static_cast<int>(reactant_mols.size());

    std::vector<int> r_counts(n);
    for (int i = 0; i < n; i++) {
        r_counts[i] = get_mapped_atom_count(*reactant_mols[i]);
    }

    int main_idx = 0;
    for (int i = 1; i < n; i++) {
        if (r_counts[i] > r_counts[main_idx]) main_idx = i;
    }

    int main_r_atoms = static_cast<int>(reactant_mols[main_idx]->getNumAtoms());
    bool main_larger = main_r_atoms >= p_count;

    float score = 0.0f;
    for (int i = 0; i < n; i++) {
        const std::string& smiles = reactant_canonical_smiles[i];
        bool in_stock_mol = in_stock.count(smiles) > 0;

        if (in_stock_mol) {
            if (!main_larger || r_counts[i] > 2) {
                score += static_cast<float>(r_counts[i]);
            }
        }

        if (!in_stock_mol) {
            if (has_element(smiles, "Mg") || has_element(smiles, "Li") || has_element(smiles, "Zn")) {
                score -= 10.0f;
            }
        }
    }

    return score;
}

float calc_rdscore(const RDKit::ROMol& p_mol, const std::vector<RDKit::ROMol*>& r_mols) {
    int p_rings = p_mol.getRingInfo()->numRings();
    int r_rings = 0;

    for (const auto* r_mol : r_mols) {
        if (!r_mol) continue;
        for (const auto& ring : r_mol->getRingInfo()->atomRings()) {
            bool has_b_si = false;
            int min_map = INT_MAX;

            for (auto idx : ring) {
                const auto* atom = r_mol->getAtomWithIdx(idx);
                const std::string& sym = atom->getSymbol();
                if (sym == "B" || sym == "Si") {
                    has_b_si = true;
                    break;
                }
                int map = atom->getAtomMapNum();
                if (map > 0 && map < min_map) min_map = map;
            }

            if (!has_b_si && min_map < 900) r_rings++;
        }
    }

    return (p_rings > r_rings) ? 1.0f : 0.0f;
}

// ============================================================================
// Match a single molecule against pre-parsed template library
// ============================================================================

struct MatchResult {
    std::unordered_map<std::string, std::tuple<float, std::string, int, float>> results;
    std::vector<int> valid_template_ids;
};

MatchResult match_single_molecule(
    const std::string& product_smiles,
    TemplateLibrary& lib,
    const MatchConfig& config
) {
    const auto& in_stock = lib.get_inventory();

    // Parse product molecule once
    std::unique_ptr<RDKit::ROMol> p_mol(RDKit::SmilesToMol(product_smiles));
    if (!p_mol) {
        return {};
    }

    // Pre-parse product as rdchiral Reactants
    rdchiral::Reactants reactants(product_smiles);

    MatchResult result;
    std::unordered_set<int> seen_ids;

    for (int idx = 0; idx < lib.size(); idx++) {
        // Use pre-parsed reaction — no SMARTS parsing overhead
        auto [products, meta] = lib.get(idx).run(reactants, true /* keep_mapnums */);

        int n_mapped = static_cast<int>(products.size());
        if (n_mapped == 0) continue;

        float mdscore = 1.0f / n_mapped;
        bool template_is_private = lib.is_private(lib.raw(idx));

        for (int j = 0; j < n_mapped; j++) {
            const std::string& r = products[j];

            std::string canonical_r = canonical_from_smiles(r);
            if (canonical_r.empty()) continue;

            if (result.results.count(canonical_r)) continue;

            std::vector<std::string> reactant_list = split_reactants(r);
            std::vector<std::unique_ptr<RDKit::ROMol>> reactant_mols;
            std::vector<std::string> reactant_canonical;
            reactant_mols.reserve(reactant_list.size());
            reactant_canonical.reserve(reactant_list.size());

            for (const auto& smi : reactant_list) {
                std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smi));
                if (!mol) continue;
                reactant_canonical.push_back(canonical_without_map(*mol));
                reactant_mols.push_back(std::move(mol));
            }

            if (reactant_mols.empty()) continue;

            std::vector<RDKit::ROMol*> raw_ptrs;
            for (const auto& m : reactant_mols) raw_ptrs.push_back(m.get());

            float rdscore = calc_rdscore(*p_mol, raw_ptrs);
            float cdscore = calc_cdscore(*p_mol, reactant_list, raw_ptrs);
            float asscore = calc_asscore(*p_mol, raw_ptrs, reactant_canonical, in_stock);

            float private_val = template_is_private
                ? config.w_private * config.private_bonus : 0.0f;

            float score = config.w_cd * cdscore
                        + config.w_as * asscore
                        + config.w_rd * rdscore
                        + config.w_md * mdscore
                        + private_val;

            result.results[canonical_r] = std::make_tuple(score, lib.raw(idx), idx, rdscore);
            if (!seen_ids.count(idx)) {
                seen_ids.insert(idx);
                result.valid_template_ids.push_back(idx);
            }
        }
    }

    return result;
}

// ============================================================================
// Single molecule match — the only entry point.
// Python handles concurrency (asyncio, ProcessPoolExecutor, etc.).
// ============================================================================

struct MatchOutput {
    py::dict results;
    py::list valid_template_ids;
};

MatchOutput match_all_templates(
    const std::string& product_smiles,
    TemplateLibrary& lib,
    const MatchConfig& config
) {
    MatchResult cpp_result = match_single_molecule(product_smiles, lib, config);

    MatchOutput output;
    for (const auto& [key, val] : cpp_result.results) {
        output.results[py::cast(key)] = py::make_tuple(
            std::get<0>(val),
            std::get<1>(val),
            std::get<2>(val),
            std::get<3>(val)
        );
    }
    for (int idx : cpp_result.valid_template_ids) {
        output.valid_template_ids.append(idx);
    }
    return output;
}

// ============================================================================
// pybind11 module
// ============================================================================

PYBIND11_MODULE(simpretro_match, m) {
    m.doc() = "SimpRetro template matching engine (C++, pre-parsed templates)";

    py::class_<TemplateLibrary>(m, "TemplateLibrary")
        .def(py::init<const std::vector<std::string>&>(),
             py::arg("templates"),
             "Pre-parse all templates into rdchiral Reaction objects (done once at init)")
        .def("__len__", &TemplateLibrary::size)
        .def("set_inventory_file", &TemplateLibrary::set_inventory_file,
             py::arg("path"),
             "Load inventory directly from file — bypasses pybind11 transfer")
        .def("set_private_templates_file", &TemplateLibrary::set_private_templates_file,
             py::arg("path"),
             "Load private templates from JSON file — bypasses pybind11 transfer");

    py::class_<MatchConfig>(m, "MatchConfig")
        .def(py::init<>())
        .def_readwrite("w_cd", &MatchConfig::w_cd)
        .def_readwrite("w_as", &MatchConfig::w_as)
        .def_readwrite("w_rd", &MatchConfig::w_rd)
        .def_readwrite("w_md", &MatchConfig::w_md)
        .def_readwrite("w_private", &MatchConfig::w_private)
        .def_readwrite("private_bonus", &MatchConfig::private_bonus);

    py::class_<MatchOutput>(m, "MatchOutput")
        .def(py::init<>())
        .def_readwrite("results", &MatchOutput::results)
        .def_readwrite("valid_template_ids", &MatchOutput::valid_template_ids);

    m.def("match_all_templates", &match_all_templates,
        py::arg("product_smiles"),
        py::arg("template_library"),
        py::arg("config") = MatchConfig(),
        "Match single product against pre-parsed template library");
}
