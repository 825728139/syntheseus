# Vue前端项目后端路由分析

## 概述
这是一个Askcos（化学合成路径规划）Vue前端项目，使用FastAPI作为后端服务。

---

## 1. 后端路由定义位置

### 核心API配置文件
**文件路径**: `src/common/api.js`

这是项目中所有API调用的统一封装层，提供了：
- `API.get(endpoint, params, query)` - GET请求
- `API.post(endpoint, data, query)` - POST请求
- `API.put(endpoint, data)` - PUT请求
- `API.delete(endpoint, data, query)` - DELETE请求
- `API.runCeleryTask(endpoint, data, progress)` - 异步任务执行
- `API.pollCeleryResult(taskId, progress)` - 任务结果轮询

### API代理配置
**文件路径**: `vite.config.js`

```javascript
const fastapiGatewayPtr = {
  target: "http://192.168.100.108:5001/",  // FastAPI后端地址
  changeOrigin: true,
  ws: true,
  secure: false,
};

server: {
  proxy: {
    "/openapi.json": fastapiGatewayPtr,
    "/docs": fastapiGatewayPtr,
    "/api/": fastapiGatewayPtr,
  },
}
```

---

## 2. 后端API路由列表

### 用户认证
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/admin/token` | POST | 用户登录获取Token |
| `/api/user/register` | POST | 用户注册 |
| `/api/user/am-i-superuser` | GET | 检查是否为超级用户 |

### 化学结构处理
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/rdkit/canonicalize/` | GET | SMILES标准化 |
| `/api/rdkit/to-molfile/` | POST | SMILES转Molfile格式 |

### 合成路径规划
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/forward/controller/call-async` | POST | 合成预测（正向） |
| `/api/retro/controller/call-async` | POST | 逆合成分析 |
| `/api/tree-search/mcts/call-async` | POST | MCTS树搜索 |
| `/api/fast-filter/call-async` | POST | 快速筛选 |
| `/api/scscore/call-sync` | GET | SCScore计算 |
| `/api/atom-map/controller/call-async` | POST | 原子映射 |

### 反应分类
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/reaction-classification/call-async` | POST | 反应分类 |

### 杂质预测
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/impurity-predictor/call-async` | POST | 杂质预测 |

### 购买信息查询
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/buyables/list-buyables` | POST | 列出可购买化学品 |
| `/api/buyables/lookup` | POST | 查找购买信息 |
| `/api/buyables/sources` | GET | 获取购买来源 |

### 结果管理
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/results/list` | GET | 获取结果列表 |
| `/api/results/destroy` | DELETE | 删除结果 |

### 任务管理
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/legacy/celery/task/{taskId}/` | GET | Celery任务状态查询 |

### 上下文推荐
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/legacy/context/` | POST | 上下文推荐 |

### 模板管理
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/template/sets/` | GET | 获取模板集 |

### 前端配置
| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/frontend-config/get-all-config` | GET | 获取所有前端配置 |

---

## 3. URL构建方式

### 构建流程
1. **前端调用**: `API.get('/api/results/list', null, false)`
2. **API封装处理** (在 `src/common/api.js`):
   ```javascript
   async request(method, endpoint, data, query = false) {
     const url = query ? `${endpoint}?${new URLSearchParams(data)}` : endpoint;
     const options = {
       method,
       headers: this.getHeaders(data),  // 自动添加Bearer Token
       credentials: "include",
     };
     const response = await fetch(url, options);
     return this.fetchHandler(response);
   }
   ```
3. **Vite代理转发**: 将 `/api/*` 请求转发到 `http://192.168.100.108:5001/`
4. **最终URL**: `http://192.168.100.108:5001/api/results/list`

### URL特点
- 所有API路径都以 `/api/` 开头
- 使用相对路径，由Vite代理处理跨域
- 查询参数使用 `URLSearchParams` 构建
- 认证Token自动从localStorage获取并添加到请求头

### 认证机制
```javascript
getHeaders(data) {
  const headers = {};
  const token = localStorage.getItem("accessToken");
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  if (data && !(data instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  return headers;
}
```

---

## 关键文件位置总结

| 文件 | 位置 |
|------|------|
| API统一封装 | `src/common/api.js` |
| Vite代理配置 | `vite.config.js` |
| API日志Store | `src/store/fastapi.js` |
| 前端配置Store | `src/store/config.js` |
