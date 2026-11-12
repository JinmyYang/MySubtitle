# MySubtitle

本地运行的音频转字幕工具。**识别、断句、导出全部在本机完成，不联网、不按次计费。**

上传音频或视频，识别一次，之后可按四种场景档位反复导出 SRT / TXT。**只支持普通话（中文）。**

- **默认引擎**：Fun-ASR-Nano（0.8B，阿里通义开源）。GPU 约 4GB 显存，没有显卡也能用 CPU（慢一些）
- **可选引擎**：阿里云百炼 Fun-ASR。需要自己提供 API Key，按音频时长计费
- **字级时间戳**：由模型原生 CTC 强制对齐给出，不是靠事后估算
- **断句不是 LLM**：一套确定性动态规划引擎，四个档位只换参数，同样输入必得同样输出

## 特性

- 拖拽上传音频（mp3 / wav / m4a / flac / ogg / aac…）或视频（mp4 / mkv / mov…），上限 1GB
- **识别与导出分离**：音频只识别一次，之后反复导出多版字幕，导出不产生任何计算费用
- **资产库**：音频、识别结果、字幕都登记为可复用资产，列表里能看到每条的"由来"
- **字幕预览**：顺资产链找到对应音频，点字幕条跳转播放，播放时高亮跟随
- **只服务本机**：要求 Host 是 `127.0.0.1` / `localhost`（挡 DNS rebinding），写操作再校验同源（挡 CSRF）
- **按需占用显存**：模型默认不加载，点识别才载入；页面上一键"释放显存"，不必关服务

## 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.10+ |
| 系统 | Windows / Linux / macOS（主要在 Windows 11 上验证） |
| 磁盘 | 模型约 2.1GB |
| GPU | 可选。CUDA 显卡约需 4GB 显存；无显卡时走 CPU |

## 安装

### 1. 基础依赖

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 本地识别依赖（推荐）

⚠️ **顺序很重要**：先装 torch / torchaudio，再装 funasr。torch 不要直接从 PyPI 装——Windows 上那是 CPU 版，装完 `torch.cuda.is_available()` 会是 `False`。按你的 CUDA 版本选官方索引：

```bash
# CUDA 12.6
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
# 纯 CPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-local.txt
```

`requirements-local.txt` 里有严格版本说明，其中 `kaldi-native-fbank` 是 funasr 的**可选但运行时必需**依赖——pip 依赖树里看不到它，缺了会在识别时报 `ImportError`。

### 3. 下载模型

```bash
python scripts/asr_local_probe.py --download-only
```

约 2.1GB，走 ModelScope，支持断点续传。下载到 `.runtime/models/`（不入版本库）。

> **⚠️ 必须用 ModelScope 的权重。** HuggingFace 上那份缺少 86 个 CTC 张量，
> 而 funasr 新版本是 fail-closed 设计：**不报错，只是静默不返回时间戳**。
> 脚本已经固定在 ModelScope 下载；手动下载时务必确认包含 `model.pt` 且能输出 `timestamps`。

### 4. 云端引擎（可选）

不用本地识别、只想调云端的话，配置环境变量 `DASHSCOPE_API_KEY`（阿里云百炼），
然后在页面的"识别引擎"里选"云端 · Fun-ASR"。按音频时长计费，约 0.79 元/小时。

## 启动

```bash
python scripts/serve.py
```

Windows 也可以直接双击 `MySubtitle.bat`（会检查环境、首次自动装基础依赖、就绪后打开浏览器）。
桌面快捷方式丢了就跑 `python scripts/make_shortcut.py` 重新生成。

> **默认端口 18080，不是 8000。** 8000 / 3000 最容易被本机其他应用占死，
> 本项目避开这两个端口。端口也没有写死：Windows 会动态保留一段 TCP 端口，
> 万一首选端口落进保留区，绑定会报 `WinError 10013`，而 `netstat` 看不到占用。
> `serve.py` 会依次尝试 18080 → 18081 → 18088 → 18090 → 28080，都不行就交给系统分配，
> 实际地址会打印在窗口里并记到 `.runtime/server.json`。
> 查保留区：`netsh interface ipv4 show excludedportrange protocol=tcp`

## 使用

1. 拖入音频，选识别引擎，点"开始识别"
2. 首次用本地引擎要等约 40 秒加载模型（阶段会显示"正在加载本地模型"），之后同一进程内复用
3. 识别完成后到"识别结果"页，选档位点"生成字幕"
4. 到"字幕"页点任意一条预览，可对着音频逐条校对

页面底部有一行模型状态，显存被占用时会出现"释放显存"按钮。识别过程中点会被拒绝，
不会打断推理。

## 性能参考

实测环境：RTX 4070 Laptop（8GB）、32GB 内存、Windows 11。

| 素材 | 音频时长 | 识别耗时 | RTF |
|---|---:|---:|---:|
| 短视频口播 | 3m41s | 24.1s | 0.109 |
| 体育解说 | 7m05s | 49.1s | 0.115 |
| 长访谈 | 23m52s | 308.8s | 0.216 |
| 长评测 | 24m59s | 325.2s | 0.217 |

- 模型首次加载约 **37 秒**，之后常驻复用；加载耗时与识别耗时分开记录，不混算
- 长音频 RTF 约为短视频的两倍：VAD 切段后只能逐段推理，批处理效率下降
- 耗时对 GPU 占用很敏感。同一份 187 秒素材，GPU 空闲时 21.5s，被其他进程抢占时 81s

## 断句设计：先约束，再优化

不是"看某段素材效果再加一条规则"，而是从字幕的物理与认知约束出发：

1. **不可切**：切点只能落在 token 边界；不跨句末标点与长静音；不在引语内部断开；
   不拆"的/地/得"及其中心语、不拆数量词、不拆成对符号。
2. **读得完**：单条有最短/最长显示时长和每秒字数（CPS）上限；太短会闪，太长会滞留。
3. **读得顺**：优先在标点或 ASR 句段边界收束；避免以虚词、连接词、量词开头或结尾。
4. **意群完整**：默认/沉浸档中，本来就放得下的意群内部是**不可达切点**；
   频繁档才允许为了短视频节奏切进意群。

满足硬约束后，用动态规划最小化代价：目标字数偏离、过短/过快惩罚、标点奖励、
跨句段合并成本。四个档位只是参数不同：

| 档位 | 目标/上限字数 | 最短停留 | 最长停留 | CPS 上限 | 保护意群 |
|---|---|---:|---:|---:|---|
| 默认 | 15 / 24 | 0.9s | 6s | 9 | 是 |
| 频繁 | 10 / 15 | 0.75s | 4.5s | 10 | 否 |
| 沉浸 | 20 / 30 | 1.3s | 7s | 9 | 是 |
| 按标点 | 不控制 | 0.02s | — | — | 纯标点直断 |

## 工作原理

```
音频 ──> 识别引擎 ──> segments（句级 text + 字级时间戳）──> 断句引擎 ──> SRT/TXT
         本地/云端                                    四档参数
```

**本地引擎**：Fun-ASR-Nano 先由 LLM 解码出带标点的文本，再用 CTC 分支对同一段音频做帧级分类，
把文本强制对齐到音频，得到每个 token 的起止时间。输出的是一份 segments 列表，与云端引擎结构完全一致。

⚠️ **接本地结果时有个坑**：CTC 的 `timestamps` 里**自带标点 token**，而 `tokenize._align_words`
消费完一个字后会自动跳过原文中的标点。两者叠加会让标点被消费两次，严格对齐必然失败，
然后**静默降级**成 jieba 比例插值——字幕照样出得来，只是时间轴变成估的。
所以 `asr_local.py` 里先滤掉纯标点 token，交给 `_align_words` 自己吸附。

**统一结构**：两条引擎都产出 `[{begin, end, text, words}]`，`begin/end/words` 均为毫秒。
断句引擎只认这个结构，不关心结果来自哪里。

**为什么时间轴准**：`tokenize.py` 把 ASR 词序列对齐成不可再分的 token，`split_base.py`
负责组装时间轴。一条字幕 = 一个 token 区间，于是"不切词 / 文本无损 / 时间精确 / 时间不重叠"
是**结构保证**，而不是靠事后断言追补。

**任务只在内存里，不跨重启恢复；真正持久的是 `assets/`。** 服务重启只重建资产索引
（音频 / 识别结果 / 字幕），所以"磁盘上有结果、内存里没任务"不会造成数据丢失。

## 目录结构

```
app/
  config.py         配置、引擎注册表、四档断句配置
  asr_local.py      本地 Fun-ASR-Nano 后端（懒加载 / 推理锁 / 卸载释放显存）
  dashscope_client  云端百炼 REST 客户端（临时上传 / 异步转写 / 轮询）
  tokenize.py       【地基】token 化：O(n) 对齐、碎片合并、成对符号回收
  split_base.py     【地基】把切分结果组装成带时间戳的字幕条
  split_punct.py    【档位】按标点：保留 ASR 标点直断
  split_presets.py  【引擎】默认/频繁/沉浸：同一套约束与 DP，只换目标参数
  export_split.py   门面：按档位分发，不放断句逻辑
  jobs.py           识别与导出任务（内存态；已完结任务会剪枝）
  assets.py         资产库：音频 / 识别结果 / 字幕的登记、索引与链条
  srt.py            SRT 生成与解析
  main.py           路由 + 只服务本机的访问控制
static/             Web 页面（原生 HTML/JS，无前端构建步骤）
scripts/            启动、探针、批量识别、评测、断句回归
tests/              无头端到端测试（mock 云端接口，不产生费用）
assets/             资产库（运行期生成，不进版本库）
.runtime/           模型与运行期状态（不进版本库）
```

## 测试

```bash
pytest tests/ -v                  # 无头端到端测试，mock 云端接口，零费用
python scripts/regress_export.py  # 断句引擎离线回归：文本无损、不超限、时间单调不重叠
```

测试用例数随本地素材增减，找不到样本时会 skip 而不是静默变成"零个用例"。

## 已知限制

- **仅普通话**。模型本身支持方言和英日语，但本项目固定按中文识别，未开放切换
- **测试素材不随仓库分发**。`测试音频/` 下的音频来自第三方视频，仅用于本地回归，版权不在本项目
- 未做说话人分离、字幕翻译、时间轴手改
- 长音频（>30 分钟）未做压力验证，理论可行但会占用较久显存

## 许可证

- **本项目代码**：[MIT](LICENSE)
- **Fun-ASR-Nano 模型权重**：独立授权，见 [ModelScope 模型页](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) 与
  [FunASR Model License](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)。商用前请自行核对
- **FunASR 工具包**：MIT

代码许可证不覆盖模型权重，分发模型前请单独确认模型许可条款。

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) 与 [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) —— 阿里通义实验室开源的语音识别工具包与模型
- [jieba](https://github.com/fxsjy/jieba) —— 中文分词，用于判断词边界
