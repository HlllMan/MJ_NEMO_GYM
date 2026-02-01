# -*- coding: utf-8 -*-
from mjnemogym.math_with_judge.app_copy import score_fn as math_fn
from mjnemogym.code_gen.app_copy import score_fn as code_fn
from mjnemogym.instruction_following.app_copy import score_fn as if_fn
from mjnemogym.structured_outputs.app_copy import score_fn as so_fn
from mjnemogym.mcqa.app_copy import score_fn as mcqa_fn
from mjnemogym.workspace_assistant.app_copy import score_fn as wa_fn


def main():
    print("🚀 正在对 NEMO_GYM 全量 6 模块进行压力测试...\n")

    # 每个模块两条：一条设计为 score=1，一条为 score=0
    # 参数格式按各 app_copy 中 score_fn 的真实签名填写
    test_suite = [
        # ----- MATH: score_fn(model_output, label) -----
        {"name": "MATH", "expect": 1, "call": lambda: math_fn(model_output=r"\boxed{5}", label="5")},
        {"name": "MATH", "expect": 0, "call": lambda: math_fn(model_output=r"\boxed{3}", label="5")},
        # ----- CODE: score_fn(model_output, verifier_unit_tests) -----
        {
            "name": "CODE",
            "expect": 1,
            "call": lambda: code_fn(
                model_output="```python\nprint(1)\n```",
                verifier_unit_tests={"inputs": [""], "outputs": ["1"]},
            ),
        },
        {
            "name": "CODE",
            "expect": 0,
            "call": lambda: code_fn(
                model_output="```python\nprint(2)\n```",
                verifier_unit_tests={"inputs": [""], "outputs": ["1"]},
            ),
        },
        # ----- IF: score_fn(model_output, instruction_id_list, kwargs) -----
        {
            "name": "IF",
            "expect": 1,
            "call": lambda: if_fn(
                model_output="<<Title>>\n\nPara one.\n\n***\n\nPara two.",
                instruction_id_list=["paragraphs:paragraphs", "detectable_format:title"],
                kwargs=[{}, None],
            ),
        },
        {
            "name": "IF",
            "expect": 0,
            "call": lambda: if_fn(
                model_output="Short.",
                instruction_id_list=["paragraphs:paragraphs", "detectable_format:title"],
                kwargs=[{}, None],
            ),
        },
        # ----- SO: score_fn(model_output, schema_str, schema_type) -----
        {
            "name": "SO",
            "expect": 1,
            "call": lambda: so_fn(
                model_output='{"a": 1}',
                schema_str='{"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}',
                schema_type="json",
            ),
        },
        {
            "name": "SO",
            "expect": 0,
            "call": lambda: so_fn(
                model_output='{"a": "x"}',
                schema_str='{"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}',
                schema_type="json",
            ),
        },
        # ----- MCQA: score_fn(model_output, label, options=...) 需传 options 才有 allowed_letters -----
        {
            "name": "MCQA",
            "expect": 1,
            "call": lambda: mcqa_fn(
                model_output=r"The answer is \boxed{A}",
                label="A",
                options=[{"A": "A"}, {"B": "B"}, {"C": "C"}, {"D": "D"}],
            ),
        },
        {
            "name": "MCQA",
            "expect": 0,
            "call": lambda: mcqa_fn(
                model_output=r"The answer is \boxed{B}",
                label="A",
                options=[{"A": "A"}, {"B": "B"}, {"C": "C"}, {"D": "D"}],
            ),
        },
        # ----- WA: score_fn(ground_truth_actions, predicted_actions) 用只读工具 company_directory_find_email_address 可成功执行 -----
        {
            "name": "WA",
            "expect": 1,
            "call": lambda: wa_fn(
                ground_truth_actions=[{"name": "company_directory_find_email_address", "arguments": "{}"}],
                predicted_actions=[{"name": "company_directory_find_email_address", "arguments": "{}"}],
            ),
        },
        {
            "name": "WA",
            "expect": 0,
            "call": lambda: wa_fn(
                ground_truth_actions=[{"name": "send_email", "arguments": "{}"}],
                predicted_actions=[{"name": "send_email", "arguments": "{}"}],
                error="mock error for score=0",
            ),
        },
    ]

    print(len(test_suite))
    for case in test_suite:
        try:
            score = case["call"]()
            ok = "✅" if score == case["expect"] else "⚠️"
            print(f"{ok} [{case['name']:<4}] 期望={case['expect']} 得分={score}")
        except Exception as e:
            print(f"❌ [{case['name']:<4}] 异常: {e}")

    print("\n🎉 若出现 score=1 和 score=0 各至少一条，且 [WA] 无 ModuleNotFoundError，说明 Agent 环境已就绪。")


if __name__ == "__main__":
    main()
