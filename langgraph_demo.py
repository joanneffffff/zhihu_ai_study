from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


"""
1. 暂停工作流 - 阻止自动执行危险操作                                                                                                                            
2. 传递信息 - 把决策信息发给人类
3. 等待恢复 - 用 Command(resume=...) 继续执行                                                                                                                   
4. 状态持久化 - 同一个 thread_id 可以随时恢复，不限于同一进程
"""

# 1. 定义全局 State（面试官常考：怎么管理上下文）
class AgentState(TypedDict):
    # 这里的 Annotated 和 operator.add 是为了让每次返回的 message 自动累加，而不是覆盖
    messages: Annotated[list, lambda x, y: x + y]
    code_result: str  # 存放 Sub-Agent 的执行结果


# 2. 定义 Sub-Agent 节点（也就是你说的“沙箱”）
def sub_agent_sandbox(state: AgentState):
    print("🤖 Sub-Agent 正在沙箱中执行高危代码...")
    # 模拟执行了一段生成代码的逻辑
    return {"code_result": "rm -rf / (高危删除指令)"}


# 3. 定义 Supervisor 节点（主 Agent，负责协调和触发中断）
def supervisor_node(state: AgentState):
    code_result = state.get('code_result', '')

    # 触发中断！把当前的危险代码抛给人类看
    # 恢复时，interrupt 会直接返回用户的决策值
    human_decision = interrupt({
        "action": "delete_files",
        "details": code_result,
        "question": "是否批准执行该高危指令？(输入 yes 或 no)"
    })

    if human_decision == "yes":
        print(f"⚠️ 已批准执行：{code_result}")
        print("🔥 危险操作已执行！（模拟）")
    else:
        print(f"❌ 已拒绝执行：{code_result}")
        print("🛡️ 操作已被安全拦截")

    return {"messages": [f"人类审批结果：{human_decision}"]}


# 4. 定义最终的 Chat Agent 节点（负责输出最终结果）
def chat_agent_node(state: AgentState):
    final_msg = f"任务结束。审批记录：{state['messages'][-1]}"
    return {"messages": [final_msg]}


# 5. 构建图（把节点连起来）
builder = StateGraph(AgentState)
builder.add_node("sub_agent", sub_agent_sandbox)
builder.add_node("supervisor", supervisor_node)
builder.add_node("chat_agent", chat_agent_node)

# 规定执行顺序：Sub-Agent -> Supervisor -> Chat-Agent
builder.add_edge(START, "sub_agent")
builder.add_edge("sub_agent", "supervisor")
builder.add_edge("supervisor", END)  # 实际场景这里可以连到 chat_agent

# 6. 编译图（注入 Checkpointer，这是实现“暂停与恢复”的底层基石）
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

# --- 下面是模拟面试时的运行与恢复流程 ---

# 配置一个唯一的 thread_id（相当于这次任务的会话ID）
config = {"configurable": {"thread_id": "interview-task-001"}}

print("=== 第一次运行：触发中断 ===")
# 第一次 invoke，工作流会在 supervisor 节点被 interrupt 拦住
result = app.invoke({"messages": ["开始执行任务"]}, config)

# 检查是否有 interrupt
if "__interrupt__" in result:
    print("工作流已安全暂停！\n")
    interrupt_data = result["__interrupt__"][0].value
    print(f"中断信息: {interrupt_data}")

    # 真正的 CLI 交互
    print("\n=== 模拟人类审批，恢复运行 ===")
    decision = input("是否批准执行该高危指令？(yes/no): ")
    final_state = app.invoke(Command(resume=decision), config)

    print("\n=== 最终 State 结果 ===")
    print(final_state)
else:
    print("未触发中断，直接完成")
    print(result)