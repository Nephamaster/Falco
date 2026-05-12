# TODO

---

# QA

## 一、项目总览

### *你用一句话介绍 FalcoAgent，它和普通 ChatBot / ReAct Agent demo 的核心区别是什么？*
  FalcoAgent 本质上不是一个单体对话模型，而是一个面向复杂任务的 Agent 系统框架。它在架构上引入了多 Agent 协作（由 secretary 负责任务路由与调度）、分层记忆体系（短期上下文 + daily + evergreen 长期记忆）、工程化 RAG 管线，以及内置的评测与 trace 机制。相比普通 ChatBot 或 ReAct demo 只是在单轮或短上下文里做“思考-调用工具-回答”，FalcoAgent 更关注任务级执行能力、长期状态管理以及系统可评估性。
  核心差异总结：
  普通 ChatBot / ReAct 的本质是“单 Agent + prompt 驱动的推理循环”，而 FalcoAgent 是“调度型 Agent 系统”，核心区别在三点：
  **从单体推理 → 多角色协作（router + worker）**
  **从无状态对话 → 分层持久化记忆**
  **从demo级调用 → 可评测、可演化的工程系统**
  加一句拔高（加分项）
  可以把 ReAct 看成 FalcoAgent 的一个子能力，而 FalcoAgent 解决的是“如何把这种能力组织成一个可长期运行、可扩展、可优化的系统”。

### *你为什么把它定位成“个人 AI 秘书”，而不是“Agent 框架”或“RAG 应用”？*
### *这个项目的核心技术难点是什么？如果只能选 3 个，你会选哪些？*
### *你简历里写“端到端智能体架构”，端到端具体指哪几层？*
### *FalcoAgent 当前最接近产品化的部分是什么？最不像产品、还偏 demo 的部分是什么？*
### *如果面向真实用户上线，你认为第一个会暴露的问题是什么？*

## 二、Agent 编排与 LangGraph

### *为什么选择 LangGraph，而不是自己写一个 while-loop ReAct？*
  因为我的目标不是写一个单轮 ReAct demo，而是做一个可恢复、可扩展、可观测的 Agent 执行框架。
  如果只是“模型思考、调用工具、再继续思考”，while-loop 确实够用。但 FalcoAgent 里面有多 Agent 调度、工具调用、RAG、记忆写入、人工介入、trace 和评测等环节，这些节点之间存在明确的状态流转关系。LangGraph 更适合把这些流程建模成 **StateGraph**：每个节点负责一个确定职责，边负责路由，状态集中管理。
  核心原因有三个。
  第一，LangGraph 对状态管理更清晰。while-loop 里上下文、工具结果、记忆、错误信息容易堆在一个大变量里，后期很难维护；LangGraph 可以把 thread state、messages、tool results、memory candidates 等结构化保存和传递。
  第二，LangGraph 天然支持中断和恢复。比如需要人工确认高风险操作、补充缺失信息，或者工具执行失败后重试，while-loop 需要自己设计 checkpoint、resume、pending state；LangGraph 的 checkpointer 和 interrupt/resume 机制更适合这类长任务 Agent。
  第三，它更利于工程扩展和调试。后续加 planner、subagent、RAG、memory postprocess，只需要增加节点和边；同时每一步执行路径都可以 trace，便于评测和定位问题。
  所以我的选择是：**ReAct loop 适合验证 Agent 思路，LangGraph 适合把 Agent 做成可长期演进的工程系统。**

### *如果工具调用失败，agent 会如何恢复？恢复逻辑主要依赖 prompt、代码还是 Reflexion？*
  工具调用失败后的恢复主要依赖 **代码层的确定性控制**，prompt 和 Reflexion 是辅助。
  可以这样回答：
  FalcoAgent 里工具失败不会完全交给模型自由发挥，而是先在代码层捕获异常，把失败类型、错误信息、工具名、参数、调用阶段写入状态和 trace。然后根据错误类别做确定性处理，比如参数缺失就转入 clarification 或 human-loop，临时性错误可以重试，权限或高风险问题进入人工确认，不可恢复错误则把失败原因返回给上层 Agent，由 Agent 重新规划。
  prompt 的作用是让模型基于结构化错误信息重新判断下一步，例如是否换工具、补充参数、降级回答或者终止任务。但它不是唯一恢复机制，否则会不稳定。
  Reflexion 更偏长期改进，不直接承担本次恢复。它会在一次失败结束后抽取可复用经验，比如“某类工具调用前必须校验路径存在”“写文件前需要确认权限”，用于后续优化 prompt、tool schema 或执行策略。
  所以总结就是：
  **本次恢复靠代码控制流和状态机，局部重规划靠 prompt，长期避免重复失败靠 Reflexion。**

### *你现在的编排是单 lead agent。什么时候需要 planner/executor 分离？*
当任务从“简单路由 + 直接执行”变成“多步骤、强依赖、需要回滚或并行”的时候，就需要 planner/executor 分离。
  我现在的单 lead agent 更适合中短任务：识别意图，选择 sub-agent 或 tool，拿到结果后汇总回答。它的优点是链路短、延迟低、实现简单。但如果用户任务变成“调研资料 → 拆解计划 → 多源检索 → 代码修改 → 运行验证 → 生成报告”这种长链路，单 agent 同时负责规划和执行就会有几个问题：容易边做边改计划，状态混乱；失败后不知道回退到哪一步；多个子任务难以并行；评测时也难以判断到底是计划错了还是执行错了。
  这时我会把 planner 和 executor 分离。planner 只负责把用户目标拆成结构化任务图，比如每一步的目标、依赖、输入输出、验收标准和失败策略；executor 只负责按计划调用工具、执行子任务、产出结果并回传状态。这样好处是计划可审查、执行可观测、失败可定位，也更容易做人类确认和并行调度。
  一句话总结就是：
  **单 lead agent 适合轻量任务编排；planner/executor 分离适合长链路、多工具、多依赖、需要可控执行的复杂任务。**

### *当前 checkpointer 是 `InMemorySaver`，如果服务重启，线程状态会发生什么？*
  如果当前 checkpointer 是 `InMemorySaver`，状态只存在进程内存里。服务一旦重启，已有 thread 的 checkpoint 会全部丢失。
  当前版本如果使用 `InMemorySaver`，它只适合本地开发和短会话调试，不适合生产持久化。服务重启后，LangGraph 无法根据原来的 `thread_id` 恢复到之前的 graph state，interrupt 后的 pending 状态、历史 messages、工具中间结果都会丢失。用户再次请求时，本质上会从一个新的空状态或外部重新构造的上下文开始执行。
  生产环境我会替换成持久化 checkpointer，比如 SQLite、Postgres 或 Redis-backed saver。这样 `thread_id + checkpoint namespace` 可以映射到稳定存储，服务重启后仍然能恢复到上一次中断或完成前的状态。
  一句话总结：
  **`InMemorySaver` 只保证进程生命周期内可恢复；跨进程、跨重启恢复必须换成持久化 checkpointer。**

### *你搭建的 `context -> agent -> tools -> persist` 状态图，每个节点的职责是什么？*
### *你的 ReAct 闭环里，“思考-行动-观察-回答”分别对应代码里的哪些对象或消息类型？*
### *`hydrate_context` 为什么要在每轮最开始做？它加载了哪些上下文？*
### *你如何控制工具调用轮数？`max_tool_steps` 的设计有没有 off-by-one 风险？*
### *如果模型在工具预算耗尽后还想继续调用工具，你的系统如何处理？*
### *如果用户任务很长，单次 LangGraph invoke 被打断后如何恢复状态？*

## 三、工具系统与安全边界

###  *你的 atomic tools 包括哪些？为什么要设计成这些粒度？*
  FalcoAgent 里的 atomic tools 主要分成几类：文件与工作区工具，例如 `list_files`、`read_file`、`write_file`、`search_in_files`、`change_working_directory`；记忆工具，例如 `add_memory`、`query_memory`；RAG 和 Skill 工具，例如 `rag_search`、`use_skill`、`skill_catalog`、`skill_manage`；人工介入工具，例如 `ask_clarification`、`approve_pending_action`、`deny_pending_request`；以及多 Agent 相关工具，例如 `delegate_task`、`run_subagent_tasks`、`read_subagent_result`、`collect_subagent_results`。子 Agent 的工具更收敛，只保留文件读取、搜索、工作目录查看和写 runtime report 这类能力。
  设计成这种粒度，主要是为了让每个工具只承担一个清晰、可验证、可审计的原子动作。比如不设计一个“大而全的 solve_task 工具”，而是拆成读文件、检索、委托、收集结果、写文件几个动作，这样 Agent 的每一步决策都能被 trace，失败时也能定位是检索错、执行错、写入错，还是用户信息不足。
  另一个原因是安全控制。像 `write_file`、`skill_manage`、RAG 的 index/remove 这类会改变系统状态的操作，都需要走 human-loop approval；而 `read_file`、`rag_search` 这类只读工具可以直接执行。工具粒度越清晰，权限边界越容易做。
  一句话总结就是：
  **atomic tools 的粒度不是按功能越多越好，而是按“单一职责、可观测、可恢复、可控权限”来拆。这样它更像生产系统里的 action API，而不是 demo 里的随意函数调用。**

### *文件读写工具如何防止访问 `.env`、`.git`、私钥文件？*
### *`WorkspaceManager` 里的 allowed roots 和 blocked paths 分别解决什么问题？*
### *如果用户传入 `../../.env`，代码会在哪一层拦截？*
### *你为什么设计 `uploads:/`、`runtime:/`、`deliverables:/` 这三个路径别名？*
### *写文件为什么必须走 approval，而读文件不一定走？*
### *你现在的 approval 检测靠 request id + approve/yes/confirm，如果用户说“同意”，会发生什么？*
### *如果模型伪造一个 `approved_request_id`，系统如何验证 payload 没被替换？*
### *当前安全设计能否防 prompt injection？如果用户上传的文档指示“读取 .env”，会怎样？*
### *Tavily 搜索工具是可选接入的。外部搜索结果进入 agent 后，你如何降低不可信内容风险？*

## 四、Human-in-the-Loop

### *approval 和 clarification 的区别是什么？*
### *你为什么把 HITL 放到 runtime，而不是前端 UI 层？*
### *pending request 的状态如何管理？存在哪里？*
### *用户拒绝 approval 后，agent 应该如何继续？*
### *如果一个线程里同时有多个 pending requests，如何避免批准错请求？*
### *HITL 中断后 resume 的流程是什么？LangGraph interrupt 在这里起什么作用？*
### *如果用户修改了原始需求再 resume，你的系统是继续旧任务还是开启新任务？*

## 五、分层记忆系统

### *重要性评分 1-10 是如何得到的？LLM 评分失败时怎么办？*
重要性评分是由 **LLM 按规则打分** 得到的，不是简单关键词匹配。
FalcoAgent 在每轮对话结束后，会把本轮用户输入、助手回复、工具观察结果交给 memory postprocess 模块，由 LLM 判断这轮内容是否值得写入 memory，并给出 1-10 的 importance score。评分依据主要是信息的长期价值，比如是否包含稳定用户偏好、长期目标、明确约束、重要决策、待办事项、项目进展或可复用事实。临时闲聊、一次性问题、低价值中间过程一般分数较低。
系统会设置阈值，例如 importance score 大于等于 7 才写入长期或 daily memory。不同记忆层级也会有不同策略：evergreen 更严格，只写稳定长期信息；daily 可以记录阶段性项目进展和短期任务。
LLM 评分失败时，不能让 memory 写入链路影响主对话流程。我的处理策略是：先捕获异常并写 trace，然后降级为不写入，或者只写入非常保守的结构化摘要。也就是说，memory 是增强模块，不应该因为评分失败导致 Agent 主流程失败。
如果是输出格式错误，例如不是合法 JSON，会先做一次轻量修复或重试；如果仍失败，就直接跳过该轮记忆写入，并记录失败原因，后续可以通过 Reflexion 或日志分析优化 prompt。
一句话总结：**重要性评分由 LLM 根据长期价值判断生成；失败时走保守降级，宁可少写，也不乱写，保证主 Agent 流程不被 memory 模块拖垮。**

### *Reflexion 如何判断一条经验值得写入？confidence 阈值为什么是 0.65？*
Reflexion 的写入不是默认发生的，而是先做一次 **写入决策（reflection decision）**，本质上也是一个受约束的 LLM 判断过程。
具体来说，它会把当前这一轮的关键信息打包进去，包括用户输入、Agent 的响应、工具调用过程以及执行结果，然后让 LLM 判断是否能抽象出一个**可复用的操作性经验**。判断标准主要有三点：
第一，看是否具有跨场景复用价值。比如“某类工具调用前必须校验参数”“RAG 查询需要先做 query rewrite”，这种能提升未来行为的经验才值得写；如果只是一次性内容或用户私有信息，不会写。
第二，看是否和执行质量直接相关。只有那些能改善 **planning、tool choice、参数构造、错误恢复** 的经验才会进入 Reflexion，而不是记录事实或对话内容，这些属于 memory 系统。
第三，看是否来源于明显的成功或失败信号。典型触发是工具失败、结果错误、走了冗余路径，或者一次特别高质量的执行路径，这些更容易被总结成“lesson”。
在实现上，我会约束 LLM 输出结构化 JSON，比如包含 `should_write`、`lesson`、`trigger`、`recommendation`、`confidence`。只有当 `should_write=true` 且置信度超过阈值时才真正写入。
如果 LLM 判断失败或输出不合规，同样走保守策略：直接不写，避免污染 Reflexion 库。
一句话总结：**Reflexion 只记录“能改变未来 Agent 行为的通用操作经验”，通过结构化决策 + 置信度过滤来控制写入质量，而不是简单记录执行过程。**

### *如何防止错误反思污染长期记忆？*
FalcoAgent 防止错误反思污染长期记忆，本质上是把 Reflexion 当成**低信任、可验证、可淘汰的弱监督信号**来处理，而不是直接当真理写入。
核心有四层防护。
第一层是**写入门控（gating）**。
Reflexion 必须满足严格条件才会写入，比如 `should_write=true` 且 `confidence` 超过阈值，同时要求内容是“可泛化的操作经验”，而不是具体案例或用户事实。如果输出格式不合法或置信度低，直接丢弃。
第二层是**作用范围隔离（scope control）**。
Reflexion 不直接写入 evergreen memory，也不会作为强约束参与主决策，而是作为“辅助提示”或“策略建议”参与后续推理。这避免了错误经验变成硬规则。
第三层是**验证与反馈（validation loop）**。
一条 Reflexion 只有在后续多次任务中被“验证有效”才会逐步提升权重；如果反复导致错误或无收益，会被降权甚至淘汰。这相当于一个在线 A/B 或信用评分机制。
第四层是**时间衰减与清理（decay & pruning）**。
所有 Reflexion 都带时间和使用统计，可以做半衰期衰减或基于命中率的清理。长期未命中或效果不佳的经验会自动被清除，防止积累噪声。
如果要一句话总结：**通过“严格写入门控 + 与核心记忆隔离 + 运行中验证反馈 + 衰减淘汰机制”，把 Reflexion 从“长期真理”降级为“可试错的策略缓存”，从而避免错误经验污染系统。**

### *如果用户偏好发生变化，比如之前喜欢简洁，现在喜欢详细，长青记忆如何更新或覆盖？*
长青记忆不能只做追加，否则会形成偏好冲突。我的处理方式是把 evergreen memory 设计成**可更新的用户画像**，而不是纯日志。
当检测到用户表达了新的稳定偏好，例如“以后回答详细一点”，memory postprocess 会先判断它是否和已有偏好冲突。如果冲突，不是简单新增一条，而是执行 **upsert / supersede**：把旧偏好标记为过期、降低权重，或者直接用新偏好覆盖对应字段，同时保留更新时间和来源轮次。
偏好更新时我会看三个信号：第一，用户是否明确使用“以后、从现在开始、以后都”这类长期表达；第二，新偏好是否被多轮重复确认；第三，新偏好是否和当前任务场景绑定。如果只是某一次任务需要详细，就写到 daily 或当前 thread，不会覆盖 evergreen。
所以“之前喜欢简洁，现在喜欢详细”这种情况，我会优先采用最近且明确的偏好，并在长期记忆里形成类似：
`response_style: detailed`
`previous: concise`
`updated_at: ...`
`scope: general / technical explanations`
一句话总结：**evergreen memory 不是无限追加，而是带冲突检测、版本更新和作用域区分的长期画像；明确的新偏好会覆盖旧偏好，临时偏好只影响当前会话。**


### *你简历里写短期上下文、每日日志、长青日记三层。为什么这样分层？*
### *短期上下文里的“关键轮次 + 全局摘要”相比只保留最近 N 轮有什么优势？*
### *重要性阈值为什么设为 7？这个值怎么调？*
### *rolling summary 如何避免不断累积错误？*
### *daily log 里记录 task/decision/constraint/preference/artifact，有什么实际作用？*
### *每日日志的 30 天半衰期公式是什么？为什么选 30 天？*
### *daily log 默认 180 天回看窗口，为什么不是全量检索？*
### *evergreen memory 为什么不做时间衰减？*
### *用户侧 evergreen 和秘书侧 Reflexion memory 为什么要分开？*
### *你的 memory retrieval 目前主要是启发式相关性排序还是向量检索？优缺点是什么？*
### *长线程场景下，memory block 如何控制 token budget？*
### *silent compaction 触发条件是什么？*
### *如果 compaction 把关键信息压没了，你有什么评估或回滚机制？*

## 六、子 Agent 委派

### *你为什么把子 agent 委派封装成工具？*
### *子 agent 和 lead agent 共享上下文吗？哪些共享，哪些隔离？*
### *子 agent 为什么必须通过 `result.md` 文件交付，而不是直接返回聊天文本？*
### *`runtime:/subagents/worker_xxx` 的文件协议包括哪些文件？*
### *子 agent 能否写最终 deliverable？为什么？*
### *如果一个子 agent 没写 `result.md`，主流程如何处理？*
### *当前子 agent 是同步执行的。如果要并行执行，你会怎么改？*
### *如何控制子 agent 数量，避免 agent 自己无限委派？*
### *子 agent 适合处理什么任务？不适合处理什么任务？*
### *多个子 agent 产出冲突时，lead agent 如何仲裁？*

## 七、RAG Skill

### *你为什么把 RAG 做成 skill，而不是内置普通 tool？*
我把 RAG 设计成 skill 而不是普通 tool，本质是因为它不是一个“原子动作”，而是一条**完整的子流程（pipeline）**。
普通 tool 更适合单步操作，比如 read_file、search、write_file，这类调用是一次输入一次输出。而 RAG 实际包含多步：query 理解与改写、检索（retrieval）、重排序（rerank）、上下文构造、再生成，有时还会有多轮检索或失败回退。如果把这些拆成多个原子 tool 让 Agent 自己拼，会导致几个问题：调用链过长、prompt 复杂度上升、决策不稳定，而且很难统一优化。
把它做成 skill，相当于把这条 pipeline 封装成一个**高层能力单元**，由 skill 内部负责流程控制和优化，对外只暴露一个稳定接口，比如“给定问题 → 返回增强后的上下文或答案”。这样有几个好处：
第一，**抽象层更清晰**。Agent 只需要决策“是否使用 RAG”，而不用关心具体怎么检索、怎么 rerank。
第二，**便于独立优化**。比如我可以单独替换 embedding 模型、调优检索策略、加入 query planner，而不影响 Agent 层。
第三，**更符合能力扩展模型**。在 Falco 里，skill 是可注册、可管理、可组合的能力单元，未来接入外部能力（MCP）时，也可以直接作为 skill 接入，而不是散落成一堆工具。
第四，**可控性更强**。skill 内部可以做缓存、失败重试、质量评估，这些如果放在 tool 层会非常碎。
一句话总结：**tool 是原子操作，skill 是流程级能力；RAG 本质是一个多阶段 pipeline，用 skill 封装比拆成多个 tool 更稳定、可控、也更容易工程化优化。**

### *RAG 的完整查询链路是什么？*
FalcoAgent 里的 RAG 是一条完整的 pipeline，从 query 到生成大致分为六步：
第一步是 **query 分析与改写**。
Agent 会先判断当前问题是否需要外部知识，如果需要，会对原始 query 做规范化或改写，比如补全上下文、消歧、或者转成更适合检索的表达。
第二步是 **向量化（embedding）**。
将 query 通过 embedding 模型编码成向量，用于后续在向量库中做相似度搜索。
第三步是 **检索（retrieval）**。
在 Milvus 中做 top-k 相似度搜索，拿到一批候选文档。这一步通常 recall 比较高，但噪声也会多。
第四步是 **重排序（rerank）**。
用 reranker（通常是 cross-encoder 或更强的 embedding）对候选文档重新打分，筛掉不相关内容，提升精度。
第五步是 **上下文构造（context building）**。
把筛选后的文档进行截断、拼接、去重，并结合 token budget 构造成 prompt context，有时会做 chunk merge 或去冗余。
第六步是 **生成（generation）**。
把 query + 构造好的 context 输入 LLM，生成最终答案。
**RAG 的本质是“query → retrieval → filtering → grounding → generation”的多阶段 pipeline，而不是简单的向量检索 + 拼 prompt。**

### *dense 检索和 BM25/sparse 检索各自适合什么问题？*
Dense 检索更适合“语义相似但字面不一致”的问题。它把 query 和文档 chunk 编成向量，用 embedding 空间里的距离找相关内容，所以对同义改写、概念匹配、上下文语义更友好。比如用户问“如何控制长期对话的上下文膨胀”，文档里写的是“token budget、rolling summary、silent compaction”，dense 更容易召回。
BM25/sparse 检索更适合“关键词精确匹配”的问题。它依赖词项重合、词频、逆文档频率，对专有名词、函数名、配置项、错误码、论文术语、字段名很敏感。比如用户问 `max_tool_steps`、`BM25BuiltInFunction`、`hitl_abc123`、某个 API 路径或具体类名，BM25 往往比 dense 稳。
所以在 FalcoAgent 里支持 hybrid 的原因是：Agent 知识库既有自然语言设计文档，也有代码、配置、接口名。dense 负责语义泛化，BM25/sparse 负责字面锚点。混合后能减少两类失败：dense 漏掉精确术语，BM25 漏掉语义改写。

### *cross-encoder rerank 为什么放在召回后？它和 embedding similarity 的区别是什么？*
Cross-encoder rerank 放在召回后，主要是因为它精度更高但成本更高。
第一阶段用 embedding similarity / dense 或 hybrid retrieval 从 Milvus 里快速召回一批候选，比如`fetch_k=18`。这一阶段适合在大规模文档库里做粗筛，因为向量可以提前离线编码，查询时只需要算 query embedding 再>做 ANN 检索，速度快、可扩展。
第二阶段才用 cross-encoder 对候选重排。Cross-encoder 会把：
```text
(query, document_chunk)
```
作为一对输入，同时送进模型，让模型直接判断这段文档和问题的相关性。它能看到 query 和 chunk 之间细粒度的 token 交>互，所以对否定、指代、局部条件、术语匹配、句间关系更敏感，相关性判断通常比单纯向量距离更准。
它和 embedding similarity 的核心区别是：
```text
embedding similarity:
query -> 向量
doc   -> 向量
再算两个向量距离
```
文档向量可以预计算，速度快，但 query 和 doc 在编码阶段彼此看不到。
```text
cross-encoder:
(query, doc) -> 一个相关性分数
```
query 和 doc 被联合编码，模型能做更细的匹配，但每个候选都要单独前向一次，不能提前给所有文档预计算分数，成本高。
所以 FalcoAgent 里采用两阶段设计：先用 dense/hybrid 召回较小候选集，再用 cross-encoder rerank top >candidates，最后输出 top-k。这样在效率和相关性之间做折中。

### *query rewrite、sub-query expansion、keyword extraction 分别解决什么问题？*
### *你如何做候选文档去重？去重 key 是什么？*
### *hybrid dense+sparse 的权重 0.7/0.3 怎么来的？*
### *Milvus Lite 不支持 BM25 sparse 时，你的系统如何降级？*
### *rerank top_n 和 final top_k 分别表示什么？*
### *chunk size 900、overlap 120 是怎么选的？*
### *如果用户上传 PDF、图片或 Word，当前 indexer 支持吗？不支持的话怎么扩展？*
### *当前 RAG 是否有 source-level 增量刷新？`refresh_source` 为什么还是 reserved？*
### *如果检索结果为空，agent 应该直接回答还是说明知识库无证据？*
### *如何评估 RAG 的召回质量和回答 groundedness？*
### *简历里写“提升召回结果相关性”，你有没有量化实验？*

## 八、MCP 动态工具

### *MCP 在 FalcoAgent 里解决了什么问题？*
MCP 在 FalcoAgent 里解决的是“外部工具动态扩展”的问题。
FalcoAgent 本身内置了文件、记忆、RAG、HITL、子 Agent 等工具，但真实个人秘书场景里，工具需求会不断变化，比如接日历、浏览器、数据库、企业系统、代码平台等。如果每接一个服务都在 FalcoAgent 里硬编码一个 tool，会导致核心 runtime 越来越臃肿，也不利于复用社区已有工具。
所以我引入 MCP，把外部能力作为标准化 tool provider 接进来。FalcoAgent 只需要读取 `.falco/mcp.json`，动态加载多个 MCP Server，然后把它们返回的 tools 归一化后追加到 LangChain tool list 里，统一交给 lead agent 调用。
所以一句话说：MCP 让 FalcoAgent 不需要改核心代码，就能接入外部工具生态，把个人秘书从“固定工具集”扩展成“可配置工具平台”。

### *你支持哪些 MCP transport？*
### *为什么要给 MCP tools 加 `mcp_<server>_<tool>` 前缀？*
### *如果两个 MCP server 暴露同名工具，系统如何处理？*
### *MCP server 配置里的 risk_level 目前实际参与安全控制了吗？*
### *MCP 工具动态 reload 的触发机制是什么？*
### *如果 MCP server 加载失败，会不会影响主 agent 启动？*
### *MCP 工具和内置工具冲突时，agent 如何选择？*

## 九、服务化与前端

### *CLI、FastAPI、Next.js 三个入口如何复用同一套核心逻辑？*
### *`/chat` 和 `/chat/stream` 的区别是什么？*
### *你的 SSE streaming 是真正模型 token streaming 吗？还是最终答案分块？*
### *前端如何识别 HITL payload 并展示 approval/clarification 卡片？*
### *thread_id 在前后端分别承担什么作用？*
### *Web 端 localStorage 存 session，有什么数据一致性问题？*
### *如果多个浏览器窗口同时操作同一个 thread，会有什么 race condition？*

## 十、代码真实性与风险追问

### *我看你的 service 层 `/api/v1/rag/search` 里访问 `orchestrator.rag`，但 orchestrator 似乎没有这个属性。这个 API 实际能跑吗？*
### *`SearchArtifacts` 没有 `render()` 方法，但 service 里调用了 `result.render()`。这是遗漏还是旧版本残留？*
### *README 写 API 支持 RAG，但代码主路径是 skill RAG。你如何解释这个不一致？*
### *你的 eval harness 目前是 offline runner，它真的能评估 agent 的长期记忆能力吗？*
### *当前 Python 文件能解析，但你有没有跑过端到端 eval？通过率是多少？*
### *你如何设计一组 eval case 来验证 HITL 机制？*
### *如果模型输出非标准 tool call，你写了 `coerce_json_tool_call`，它解决了什么 provider 兼容问题？*
### *MiniMax tool call 的 XML-like 格式如何解析？为什么要做这层兼容？*
### *如果 LLM 输出恶意 JSON tool call，valid_tool_names 能拦住什么，拦不住什么？*
### *你项目里哪些部分是工程 glue code，哪些部分真正体现算法能力？*

## 十一、算法岗深挖

### *你认为 FalcoAgent 中最有算法含量的模块是 memory、RAG、还是 Reflexion？为什么？*
我会回答：**最有算法含量的是 memory 系统，其次是 RAG，Reflexion 更像 memory 的一个高阶子模块。**
原因是 memory 不是简单保存历史对话，而是在做一个“长上下文压缩与检索”的问题：有限 token budget 下，系统要决定哪>些信息保留、哪些压缩、哪些按需召回。这里涉及多种排序和筛选策略：
- 短期记忆：recent window + global summary + key turns。
- key turns：用 LLM 对历史轮次做 1-10 重要性评分，再结合 query relevance 选择。
- daily log：结构化记录任务、决策、约束、偏好、产物，并用重要性、时间衰减和 query 相关性综合排序。
- evergreen memory：对长期偏好和反思经验不做时间衰减，只按重要性和相关性注入。
- silent compaction：当上下文接近上限时，自动压缩并沉淀高价值信息。
这本质上是在做 **记忆写入决策、长期信息保留、检索排序、上下文预算分配**，比单纯 RAG 更贴近 agent 长期运行的核心>问题。
RAG 也有算法含量，主要体现在 query rewrite、sub-query expansion、dense/BM25 hybrid retrieval、>cross-encoder rerank。但这条链路相对标准，是比较成熟的两阶段检索范式。
Reflexion 的价值在于让 agent 从工具观测和失败经验中提炼可复用策略，但目前实现上主要依赖 LLM 结构化抽取和 confidence 阈值，算法深度还没有 memory 主系统强。它更像是 evergreen memory 的一种来源，负责写入“秘书侧长期经>验”。
所以如果面试官追问，我会强调：**FalcoAgent 的核心不是某个单点检索算法，而是围绕 agent 长期任务执行设计了一套分>层记忆与上下文选择机制。**

### *memory importance scoring 能否训练一个小模型替代 LLM？需要什么数据？*
### *daily log 的排序函数如果形式化，你会如何写？*
### *query relevance 当前如果是词面启发式，如何升级成 embedding retrieval？*
### *RAG 中 query planning 如果产生错误子查询，如何检测和抑制？*
### *reranker 失败时 fallback 到原始召回，是否会引入质量波动？*
### *如何做 online learning，让 agent 的 Reflexion 真的提升任务成功率？*
### *如何证明 Reflexion 写入的经验在后续任务中被有效使用？*
### *如果长期记忆冲突，应该用 recency、confidence、importance 还是用户显式覆盖来仲裁？*
### *你会如何把 FalcoAgent 改造成企业知识助手？*

## 十二、开放设计题

### *如果让你把 FalcoAgent 上线给 1000 个用户，你第一周会改哪些模块？*
### *如何做权限系统，让不同用户只能访问自己的 workspace？*
### *如何支持真正异步、长任务、可取消、可恢复的 agent 执行？*
### *如何给每次工具调用做审计日志和可视化 trace？*
### *如何做 cost control，限制 token、工具次数、外部 API 调用？*
### *如果用户要求 agent 修改代码并自动提交 PR，你会怎样设计安全链路？*
### *如何设计 FalcoAgent 的 benchmark，和 OpenAI Codex、Claude Code、Cursor Agent 比较？*
### *如果模型能力下降或换模型，哪些模块最容易受影响？*
### *这个项目未来要发表或作为实习项目展示，你会补哪些实验？*
### *如果只能重构一个模块，你会重构哪里，为什么？*

## GPT提问

### Agent架构设计

#### *你这个 secretary + subagent 的设计，本质上和 ReAct + tool 有什么本质区别？*
本质区别在于：**ReAct + tool 通常是单 agent 在一个上下文里顺序调用工具；FalcoAgent 的 secretary + subagent >把“委派”本身做成了一个有隔离上下文、落盘协议和结果回收机制的执行单元。**
普通 ReAct + tool 里，tool 更像一次函数调用：
```text
agent -> call tool -> get observation -> continue reasoning
```
工具通常是无状态、短生命周期的，返回值直接塞回当前对话上下文。复杂任务即使拆分，也还是主 agent 自己在同一个上下>文里连续完成。
FalcoAgent 里的 subagent 不是普通函数式工具。主 secretary 调用 `delegate_task` 时，会创建一个独立 worker 目>录，写入 `task.md`、`status.json`、`result.md`，然后由 `SubAgentRunner` 启动一个隔离的 worker agent。>worker 有自己的 system prompt、自己的工具子集、自己的 max steps，并且不能直接把结果通过聊天返回，必须写到 >`result.md`。主 secretary 后续再通过 `run_subagent_tasks` 和 `read_subagent_result` 回收结果。
所以差异主要有四点：
1. **上下文隔离**  
  ReAct 工具调用共享主 agent 上下文；subagent 有独立消息上下文，只接收明确任务和必要环境信息。
2. **任务协议化**  
  ReAct tool 返回一次 observation；subagent 有固定文件协议，包含任务、状态、结果和 artifacts，便于追踪和恢>复。
3. **责任边界更清晰**  
  secretary 负责规划、拆分、集成和最终回答；subagent 只负责局部子任务，不能问用户、不能写最终 deliverable。
4. **更接近并行/可审计执行**  
  虽然当前实现是同步跑 pending tasks，但设计上每个 worker 是独立任务单元，天然适合扩展成并行、失败重试、结果审>计。
所以我不会说它完全脱离 ReAct。它底层仍然用了 ReAct 风格的 tool loop，但在“工具”之上增加了一个 agent-level >delegation abstraction。也就是说，**secretary + subagent 是 ReAct 工具调用的结构化扩展，把一次工具调用升级成>可隔离、可追踪、可回收的子任务执行过程。**

#### *你的 Agent 是如何做“任务终止判断”的？*
FalcoAgent 的任务终止判断主要不是单独训练一个 stop classifier，而是由 **LangGraph 路由逻辑 + 模型 tool >call 行为 + 工具预算** 共同决定。每一轮 lead agent 输出后，系统看两件事：
1. **模型是否还发起 tool call**  
  如果最后一条 `AIMessage` 里有 `tool_calls`，说明 agent 认为任务还没完成，需要继续执行工具，于是路由到 >`tools`。
2. **工具调用预算是否耗尽**  
  如果已有 `ToolMessage` 数量达到 `max_tool_steps - 1`，无论模型是否还想继续调用工具，都强制进入 `persist`，>让 agent 基于已有观察生成最终回答。
如果模型没有 tool call，而是直接输出自然语言回答，就认为本轮任务完成，进入 `persist`，做记忆沉淀，然后结束。
所以它的终止条件可以概括为：
```text
无工具调用 => 终止
工具预算耗尽 => 强制终止
HITL interrupt => 暂停，等待用户 resume
```
另外还有一种特殊终止：如果工具返回 approval 或 clarification payload，`handle_hitl` 会触发 LangGraph >`interrupt`。这不是完成，而是“挂起任务”，等待用户审批或补充信息。用户 resume 后，图继续执行。
这种设计的优点是简单稳定：不依赖额外模型判断，和 ReAct 的自然行为一致。但局限也明显：如果模型过早回答，系统会认>为任务结束；如果模型反复调用工具，只能靠预算截断。因此后续可以加更显式的 stop criteria，例如任务 checklist、目>标满足度评分、final answer verifier，或者针对复杂任务引入 planner 状态机。

#### *daily memory 的“30天半衰期”具体怎么实现？排序如何融合 importance + time?*
`daily memory` 的 30 天半衰期是在召回排序时实现的，不是在写入时删除旧记忆。
排序融合发生在 daily log 召回函数里。每条 daily record 会计算：
```text
score = normalized_importance * time_decay + 0.35 * query_relevance
```
其中：
```text
normalized_importance = importance / 10
time_decay = 0.5 ^ (age_days / 30)
query_relevance = 命中的 query token 数 / query token 总数
```
`importance` 负责衡量这条记忆本身是否重要，范围是 1 到 10；`time_decay` 让旧的 daily memory 逐渐降权；>`query_relevance` 保证即使一条记忆比较旧，只要和当前问题强相关，仍然有机会被召回。
举个例子：
```text
A: importance=8, age=30天, relevance=0.2
score = 0.8 * 0.5 + 0.35 * 0.2
      = 0.4 + 0.07
      = 0.47
B: importance=5, age=0天, relevance=0.8
score = 0.5 * 1.0 + 0.35 * 0.8
      = 0.5 + 0.28
      = 0.78
```
所以 B 虽然重要性低一些，但因为更新、更相关，会排在 A 前面。
面试里可以总结成一句话：**daily memory 的召回不是单纯按时间，也不是单纯按重要性，而是用“重要性 × 时间衰减 + >query 相关性”的线性融合排序；30 天半衰期通过指数衰减实现，保证旧信息平滑降权而不是硬删除。**

#### *Reflexion 写错了怎么办？有没有可能越学越差？*
有可能。Reflexion 本质上是把一次执行经验沉淀成长期策略，如果写入的是错误归因、过度泛化的经验，后续 agent 在规划>或工具选择时反复看到它，就可能越学越差。
我现在的实现做了几层防护，但还不算完美。
第一层是 **写入门槛**。Reflexion 不是每轮都写，而是基于用户输入、assistant 输出和工具观测，让 LLM 结构化判断：
```text
should_write
lesson
trigger
recommendation
confidence
tags
```
只有 `should_write=true`、`lesson` 非空，并且 `confidence >= 0.65` 才会写入 evergreen memory。
第二层是 **只写高层策略，不写具体事实**。Reflexion 更适合沉淀类似“工具失败后要检查 observation，而不是重复调用>同一工具”这种执行经验，而不是沉淀具体业务事实。这样可以降低错误事实污染长期用户记忆的风险。
第三层是 **和用户侧记忆分模块存储**。evergreen memory 分成 user 模块和 agent_reflections 模块。用户偏好、目>标、约束和 agent 自己的反思经验是分开的，避免错误 Reflexion 覆盖用户事实。
第四层是 **召回时仍然按相关性和重要性排序**。写进 evergreen 不代表每轮都注入上下文，只有和当前 query 有一定相>关性、重要性较高的 entry 才更可能进入 memory block。
如果要继续优化，我会加一个 **memory governance** 层：
- 每条 Reflexion 保留 `confidence`、`source_turn_id`、`trigger`、`success/failure_count`。
- 后续任务如果引用了某条 Reflexion，并且结果失败，就给它降权。
- 如果多次失败，标记为 stale 或 disabled。
- 对高影响 Reflexion 做二次验证，比如让另一个 judge 判断是否过度泛化。
- 提供用户侧记忆管理界面，允许删除或 pin 某些经验。
所以面试时我会这样总结：**Reflexion 确实存在越学越差的风险；当前系统通过写入阈值、模块隔离、相关性召回来缓解，但>还没有完整的闭环验证和回滚机制。真正生产化时必须加入记忆治理，否则长期自学习会变成长期污染。**

#### *MCP 和你自己写 tool 有什么本质区别？*
本质区别是：**自己写 tool 是把能力硬编码进 FalcoAgent；MCP 是把外部能力作为标准协议动态接入。**
自己写 tool 的模式是：
```text
FalcoAgent 代码里定义 tool 函数
-> 注册到 LangChain tools
-> agent 调用
```
优点是控制力强，安全边界、参数校验、权限逻辑都可以深度定制。比如文件读写、HITL、memory、subagent 这些核心能力，>我更倾向于自己写，因为它们和 FalcoAgent 的 runtime 状态强绑定。
MCP 的模式是：
```text
外部 MCP Server 暴露工具
-> FalcoAgent 读取 .falco/mcp.json
-> 通过 MCP client 动态加载工具
-> 归一化后注入 agent tool pool
```
它解决的是扩展性和生态问题。比如接日历、数据库、浏览器、GitHub、飞书、企业内部系统，不需要每次改 FalcoAgent 主>代码，只要配置对应 MCP Server，就能把外部工具接进来。
在 FalcoAgent 里，我的划分是：**核心 runtime 能力自己写 tool，外部业务能力走 MCP。**
比如：
```text
自己写 tool:
文件系统、memory、HITL、skill、RAG、subagent
MCP:
日历、浏览器、GitHub、数据库、企业系统、第三方服务
```
一句话总结：**自定义 tool 解决核心能力的可控性，MCP 解决外部能力的可插拔扩展。**