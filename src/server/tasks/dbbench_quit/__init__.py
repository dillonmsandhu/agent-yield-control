import json
import re
from typing import Callable, Dict, List, Any

from src.server.task import Task, Session
from src.typings import TaskOutput, SampleStatus, AgentOutputStatus
from .Interaction import Container
from ..dbbench import build_init_sql, DBBench

# Add quit prompt at the end:
big_prompt = """
I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose to operate or to answer.
your operation should be like this:
Action: Operation
```sql
SELECT * FROM table WHERE condition;
```
You MUST put SQL in markdown format without any other comments. Your SQL should be in one line.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If you are done operating, and you want to commit your final answer, then write down:
Action: Answer
Final Answer: ["ANSWER1", "ANSWER2", ...]
DO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
Your input will be raw MySQL response, you have to deal with it by yourself.
It is also possible to quit the task using "Action: quit". 
It is preferable to complete the task, but if you will not be able to, quit as early as possible. 
Specifically, you must maximize S - N/T, where S=1 in the case of a success and 0 in the case of failure, and N/T is the number of steps you take before quitting or succeeding normalized by the budget T=15. 
Thus, it is preferable to quit over failing.
"""


class DBBenchQuit(DBBench):
    def __init__(self, **configs):
        super().__init__(**configs)

    async def start_sample(self, index: int, session: Session) -> TaskOutput:
        entry = self.dataset[index][0]
        try:
            correct_answer = self.dataset[index][1]
        except:
            correct_answer = ""
        container = self.container
        init_sql, init_data = build_init_sql(entry)
        container.execute(init_sql, data=init_data)
        db = entry["table"]["table_name"]
        session.inject({"role": "user", "content": big_prompt})
        session.inject({"role": "agent", "content": "Ok."})
        prompt = entry["description"] + "\n" + entry["add_description"]
        session.inject({"role": "user", "content": prompt})
        res = (await session.action()).content or ""
        answer = ""
        quit = False
        finish_reason = SampleStatus.COMPLETED
        try:
            action = re.search(r"Action: (.*?)\n", res)
            rounds = 0
            while action and action.group(1) == "Operation" and rounds < self.max_round:
                # record when the agent quits and end the task
                if quit:
                    finish_reason = SampleStatus.QUIT
                    print("Breaking due to quit action")
                    break
                res = re.search(r"```sql\n([\s\S]*?)\n```", res)
                if not res:
                    finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
                    break
                sql = res.group(1).strip()
                sql = sql.replace("\n", " ")
                response = container.execute(sql, db)
                if response:
                    session.inject({"role": "user", "content": response})
                else:
                    session.inject({"role": "user", "content": ""})
                res = await session.action()
                if res.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                    finish_reason = SampleStatus.AGENT_CONTEXT_LIMIT
                    break
                res = res.content
                action = re.search(r"Action: (.*?)\n", res)
                # quit = re.search(r"Action: quit", res) is not None
                quit = "quit" in res
                rounds += 1
            else:
                answer = re.search(r"\nFinal Answer:(.*)", res)
                quit = "quit" in res
                if quit:
                    finish_reason = SampleStatus.QUIT
                elif answer:
                    answer = answer.group(1)
                else:
                    answer = ""
                    finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
                if rounds >= self.max_round and not answer:
                    finish_reason = SampleStatus.TASK_LIMIT_REACHED
        except Exception as e:
            print("Raising exception.... ", e)
            error = str(e)
            answer = ""
            finish_reason = SampleStatus.UNKNOWN
        else: # execute if no exception
            error = ""
        if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
            columns = ",".join(
                [
                    f"`{column['name']}`"
                    for column in entry["table"]["table_info"]["columns"]
                ]
            )
            md5_query = (
                f"select md5(group_concat(rowhash order by rowhash)) as hash "
                f"from( SELECT substring(MD5(CONCAT_WS(',', {columns})), 1, 5) AS rowhash FROM `{db}`) as sub;"
            )
            answer = container.execute(md5_query, db)
        container.execute(f"drop database `{db}`")
        result={
                "answer": str(answer),
                "correct_answer": correct_answer,
                "correct": answer == correct_answer,
                "type": entry["type"][0],
                "error": error,
                "description": entry["description"],
            }
        result['correct'] = self.check_answer(entry["type"][0], result, correct_answer)
        return TaskOutput(status=finish_reason, result=result, history=session.history)
