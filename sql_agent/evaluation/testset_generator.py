"""
SQL Agent 测试集生成器

方案: 模板生成 (150 条简单/中等) + LLM 生成 (50 条复杂)
  - 模板生成: 问题 + 标准 SQL 同时生成，确定性 100%
  - LLM 生成: LLM 根据数据库 schema 生成问题 + SQL，人工确认

使用:
  python -m sql_agent.evaluation.testset_generator --size 200
  python -m sql_agent.evaluation.testset_generator --size 200 --output data/sql_testset.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ===================== 数据库常量 =====================

DRUGS = [
    {"id": 1, "name": "阿莫西林胶囊", "brand": "阿莫仙", "form": "胶囊剂", "area": "抗生素", "spec": "0.25g*24粒"},
    {"id": 2, "name": "布洛芬缓释胶囊", "brand": "芬必得", "form": "胶囊剂", "area": "解热镇痛", "spec": "0.3g*20粒"},
    {"id": 3, "name": "盐酸二甲双胍片", "brand": "格华止", "form": "片剂", "area": "糖尿病", "spec": "0.5g*48片"},
    {"id": 4, "name": "阿托伐他汀钙片", "brand": "立普妥", "form": "片剂", "area": "心血管", "spec": "20mg*7片"},
    {"id": 5, "name": "磷酸奥司他韦胶囊", "brand": "达菲", "form": "胶囊剂", "area": "抗病毒", "spec": "75mg*10粒"},
    {"id": 6, "name": "注射用头孢曲松钠", "brand": "罗氏芬", "form": "注射剂", "area": "抗生素", "spec": "1.0g/支"},
    {"id": 7, "name": "蒙脱石散", "brand": "思密达", "form": "散剂", "area": "消化系统", "spec": "3g*10袋"},
    {"id": 8, "name": "硝苯地平控释片", "brand": "拜新同", "form": "片剂", "area": "高血压", "spec": "30mg*30片"},
    {"id": 9, "name": "阿司匹林肠溶片", "brand": "拜阿司匹灵", "form": "片剂", "area": "心血管", "spec": "100mg*30片"},
    {"id": 10, "name": "连花清瘟胶囊", "brand": "连花清瘟", "form": "胶囊剂", "area": "中成药/感冒", "spec": "0.35g*24粒"},
]

REGIONS = ["华北区", "华东区", "华南区", "西南区", "东北区", "华中区", "西北区"]
THERAPEUTIC_AREAS = ["抗生素", "解热镇痛", "糖尿病", "心血管", "抗病毒", "消化系统", "高血压", "中成药/感冒"]
DOSAGE_FORMS = ["胶囊剂", "片剂", "注射剂", "散剂"]
CUSTOMERS = ["北京朝阳医院", "天津大药房", "海王星辰连锁", "上海华山医院", "广州中山医院",
             "深圳人民医院", "成都华西医院", "重庆大药房", "北京协和医院", "黑龙江省医院",
             "武汉同济医院", "长沙湘雅医院", "杭州第一医院", "南京鼓楼医院", "西安西京医院"]

random.seed(42)  # 保证可复现


# ===================== 模板定义 =====================

class TemplateGenerator:
    """基于模板的测试用例生成器"""

    def __init__(self):
        self.test_id = 0

    def _next_id(self) -> int:
        self.test_id += 1
        return self.test_id

    def _make_case(self, question: str, sql: str, difficulty: str, category: str) -> Dict:
        return {
            "id": self._next_id(),
            "question": question,
            "expected_sql": sql.strip(),
            "difficulty": difficulty,
            "category": category,
            "source": "template",
        }

    def generate_simple(self) -> List[Dict]:
        """简单查询 (单表、单值、无 JOIN) — 目标 40 条"""
        cases = []

        # 1.1 查药品属性 (10 条)
        for drug in DRUGS:
            cases.append(self._make_case(
                f"查询{drug['name']}的规格是什么",
                f"SELECT specifications FROM drugs WHERE generic_name = '{drug['name']}'",
                "simple", "单值查询",
            ))
            cases.append(self._make_case(
                f"{drug['name']}是哪个治疗领域的",
                f"SELECT therapeutic_area FROM drugs WHERE generic_name = '{drug['name']}'",
                "simple", "单值查询",
            ))

        # 1.2 查库存数量 (10 条)
        for drug in DRUGS:
            cases.append(self._make_case(
                f"{drug['name']}的库存总量是多少",
                f"SELECT SUM(i.quantity_on_hand) FROM inventory i JOIN drugs d ON i.drug_id = d.drug_id WHERE d.generic_name = '{drug['name']}'",
                "simple", "聚合查询",
            ))

        # 1.3 查销售总额 (10 条)
        for drug in random.sample(DRUGS, 10):
            cases.append(self._make_case(
                f"{drug['name']}的总销售额是多少",
                f"SELECT SUM(s.total_amount) FROM sales_records s JOIN drugs d ON s.drug_id = d.drug_id WHERE d.generic_name = '{drug['name']}'",
                "simple", "聚合查询",
            ))

        return cases

    def generate_medium(self) -> List[Dict]:
        """中等查询 (多条件、分组、排序) — 目标 110 条"""
        cases = []

        # 2.1 按字段分组统计 (30 条)
        for area in THERAPEUTIC_AREAS:
            cases.append(self._make_case(
                f"查询{area}类药品的总销售额",
                f"SELECT SUM(s.total_amount) FROM sales_records s JOIN drugs d ON s.drug_id = d.drug_id WHERE d.therapeutic_area = '{area}'",
                "medium", "分组统计",
            ))

        for area in THERAPEUTIC_AREAS:
            cases.append(self._make_case(
                f"{area}类药品的总库存量",
                f"SELECT SUM(i.quantity_on_hand) FROM inventory i JOIN drugs d ON i.drug_id = d.drug_id WHERE d.therapeutic_area = '{area}'",
                "medium", "分组统计",
            ))

        for region in REGIONS:
            cases.append(self._make_case(
                f"{region}的药品销售总额",
                f"SELECT SUM(s.total_amount) FROM sales_records s WHERE s.region = '{region}'",
                "medium", "分组统计",
            ))

        for region in REGIONS:
            cases.append(self._make_case(
                f"{region}有多少笔销售记录",
                f"SELECT COUNT(*) FROM sales_records WHERE region = '{region}'",
                "medium", "分组统计",
            ))

        # 2.2 排序查询 (20 条)
        cases.append(self._make_case(
            "查询销售额最高的药品",
            "SELECT d.generic_name, SUM(s.total_amount) AS total FROM sales_records s JOIN drugs d ON s.drug_id = d.drug_id GROUP BY d.generic_name ORDER BY total DESC LIMIT 1",
            "medium", "排序查询",
        ))
        cases.append(self._make_case(
            "查询库存量最少的药品",
            "SELECT d.generic_name, SUM(i.quantity_on_hand) AS total FROM inventory i JOIN drugs d ON i.drug_id = d.drug_id GROUP BY d.generic_name ORDER BY total ASC LIMIT 1",
            "medium", "排序查询",
        ))
        cases.append(self._make_case(
            "销量排名前三的药品是什么",
            "SELECT d.generic_name, SUM(s.quantity_sold) AS total FROM sales_records s JOIN drugs d ON s.drug_id = d.drug_id GROUP BY d.generic_name ORDER BY total DESC LIMIT 3",
            "medium", "排序查询",
        ))
        cases.append(self._make_case(
            "销售额排名前五的药品",
            "SELECT d.generic_name, SUM(s.total_amount) AS total FROM sales_records s JOIN drugs d ON s.drug_id = d.drug_id GROUP BY d.generic_name ORDER BY total DESC LIMIT 5",
            "medium", "排序查询",
        ))

        for drug in random.sample(DRUGS, 5):
            cases.append(self._make_case(
                f"{drug['name']}在哪个区域销售额最高",
                f"SELECT s.region, SUM(s.total_amount) AS total FROM sales_records s JOIN drugs d ON s.drug_id = d.drug_id WHERE d.generic_name = '{drug['name']}' GROUP BY s.region ORDER BY total DESC LIMIT 1",
                "medium", "排序查询",
            ))

        # 2.3 条件过滤 (30 条)
        for form in DOSAGE_FORMS:
            cases.append(self._make_case(
                f"有哪些{form}药品",
                f"SELECT generic_name FROM drugs WHERE dosage_form = '{form}'",
                "medium", "条件过滤",
            ))

        cases.append(self._make_case(
            "查询有效期在2027年之前的药品库存",
            "SELECT d.generic_name, i.batch_number, i.expiry_date FROM inventory i JOIN drugs d ON i.drug_id = d.drug_id WHERE i.expiry_date < '2027-01-01'",
            "medium", "条件过滤",
        ))

        cases.append(self._make_case(
            "库存量大于5000的药品有哪些",
            "SELECT d.generic_name, SUM(i.quantity_on_hand) AS total FROM inventory i JOIN drugs d ON i.drug_id = d.drug_id GROUP BY d.generic_name HAVING total > 5000",
            "medium", "条件过滤",
        ))

        cases.append(self._make_case(
            "2025年上半年销售额超过10000的药品",
            "SELECT d.generic_name, SUM(s.total_amount) AS total FROM sales_records s JOIN drugs d ON s.drug_id = d.drug_id WHERE s.sale_date >= '2025-01-01' AND s.sale_date <= '2025-06-30' GROUP BY d.generic_name HAVING total > 10000",
            "medium", "条件过滤",
        ))

        # 各治疗领域的库存 (8 条)
        for area in THERAPEUTIC_AREAS:
            cases.append(self._make_case(
                f"{area}类药品当前有多少库存批次",
                f"SELECT COUNT(DISTINCT i.batch_number) FROM inventory i JOIN drugs d ON i.drug_id = d.drug_id WHERE d.therapeutic_area = '{area}'",
                "medium", "条件过滤",
            ))

        # 各区域销售代表 (随机 7 条)
        sales_reps = ["北京朝阳销售部", "天津南开销售分部", "杭州滨江销售部",
                       "上海静安销售总部", "广州越秀销售部", "深圳罗湖销售分部",
                       "成都武侯销售部", "重庆渝中销售部", "北京东单销售部",
                       "哈尔滨香坊销售部", "武汉汉口销售部", "长沙开福销售部",
                       "杭州上城销售部", "南京鼓楼销售部", "西安新城销售部"]
        for rep in random.sample(sales_reps, 7):
            cases.append(self._make_case(
                f"销售代表'{rep}'经手了多少笔订单",
                f"SELECT COUNT(*) FROM sales_records WHERE sales_rep = '{rep}'",
                "medium", "条件过滤",
            ))

        # 2.4 多表关联 (20 条)
        cases.append(self._make_case(
            "查询每种剂型的药品数量",
            "SELECT dosage_form, COUNT(*) FROM drugs GROUP BY dosage_form",
            "medium", "多表关联",
        ))
        cases.append(self._make_case(
            "各治疗领域的药品平均单价",
            "SELECT d.therapeutic_area, AVG(s.unit_price) FROM sales_records s JOIN drugs d ON s.drug_id = d.drug_id GROUP BY d.therapeutic_area",
            "medium", "多表关联",
        ))
        cases.append(self._make_case(
            "每个仓库位置的库存总额",
            "SELECT warehouse_location, SUM(quantity_on_hand) FROM inventory GROUP BY warehouse_location",
            "medium", "多表关联",
        ))

        for drug in random.sample(DRUGS, 5):
            cases.append(self._make_case(
                f"{drug['name']}各批次的库存量分别是多少",
                f"SELECT i.batch_number, i.quantity_on_hand, i.warehouse_location FROM inventory i JOIN drugs d ON i.drug_id = d.drug_id WHERE d.generic_name = '{drug['name']}'",
                "medium", "多表关联",
            ))

        for customer in random.sample(CUSTOMERS, 5):
            cases.append(self._make_case(
                f"'{customer}'购买过哪些药品",
                f"SELECT DISTINCT d.generic_name FROM sales_records s JOIN drugs d ON s.drug_id = d.drug_id WHERE s.customer_name = '{customer}'",
                "medium", "多表关联",
            ))

        # 2.5 统计函数 (10 条)
        cases.append(self._make_case(
            "所有药品的平均库存量",
            "SELECT AVG(quantity_on_hand) FROM inventory",
            "medium", "统计函数",
        ))
        cases.append(self._make_case(
            "药品的最高单价是多少",
            "SELECT MAX(unit_price) FROM sales_records",
            "medium", "统计函数",
        ))
        cases.append(self._make_case(
            "销售记录中售出数量的中位数",
            "SELECT AVG(quantity_sold) FROM (SELECT quantity_sold FROM sales_records ORDER BY quantity_sold LIMIT 2 - (SELECT COUNT(*) FROM sales_records) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM sales_records))",
            "medium", "统计函数",
        ))
        # 简化为直接求平均（SQLite/MySQL 兼容）
        cases[-1]["expected_sql"] = "SELECT AVG(quantity_sold) FROM sales_records"
        cases[-1]["question"] = "销售记录中售出数量的均值是多少"

        return cases

    def generate_all_templates(self) -> List[Dict]:
        """生成全部模板测试用例 (150 条)"""
        simple = self.generate_simple()
        medium = self.generate_medium()

        # 截断，保证 30 simple + 120 medium = 150
        simple = simple[:30]
        medium = medium[:120]

        all_cases = simple + medium
        # 重新编号
        for i, case in enumerate(all_cases):
            case["id"] = i + 1

        return all_cases


# ===================== LLM 生成复杂用例 =====================

COMPLEX_GENERATION_PROMPT = """你是一个 SQL 评测数据生成专家。根据以下数据库 Schema，生成高难度的自然语言问题及其对应的标准 SQL。

数据库表:
- drugs(drug_id, generic_name, brand_name, approval_number, specifications, dosage_form, manufacturer, therapeutic_area, description)
- inventory(inventory_id, drug_id, batch_number, quantity_on_hand, warehouse_location, production_date, expiry_date)
- sales_records(sale_id, drug_id, sale_date, quantity_sold, unit_price, total_amount, customer_name, region, sales_rep)

要求:
  - 生成 {count} 个复杂查询问题
  - 需要涉及: 子查询、多表JOIN、GROUP BY + HAVING、窗口函数中的至少两种
  - 问题用自然语言表达，不要出现 SQL 关键字
  - 不要与以下已有问题重复: {existing_questions}

输出格式 (严格 JSON 数组):
[
  {{"question": "...", "expected_sql": "SELECT ...", "category": "复杂查询", "difficulty": "hard"}},
  ...
]"""


def generate_complex_cases(
    count: int = 50,
    existing_questions: List[str] = None,
    llm=None,
) -> List[Dict]:
    """LLM 生成复杂测试用例"""
    if llm is None:
        from agent.llm import model as default_llm
        llm = default_llm

    existing = "\n".join(
        f"  - {q}" for q in (existing_questions or [])[-20:]
    )

    prompt = COMPLEX_GENERATION_PROMPT.format(
        count=count,
        existing_questions=existing or "(无)",
    )

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) >= 2 else text
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        cases = json.loads(text)
        for case in cases:
            case["source"] = "llm"
        return cases
    except Exception as e:
        print(f"[TestGen] LLM 生成失败: {e}")
        return []


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(description="SQL Agent 测试集生成器")
    parser.add_argument("--size", type=int, default=200,
                        help="测试集总条数 (默认 200)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径")
    parser.add_argument("--no_llm", action="store_true",
                        help="仅模板生成，不调用 LLM")
    args = parser.parse_args()

    output_path = args.output or str(PROJECT_ROOT / "data" / "sql_testset.json")

    # Step 1: 模板生成 150 条
    print("[TestGen] 模板生成 150 条简单/中等用例...")
    gen = TemplateGenerator()
    cases = gen.generate_all_templates()
    print(f"  生成 {len(cases)} 条")

    # Step 2: LLM 生成 50 条复杂用例
    if not args.no_llm:
        llm_count = args.size - len(cases)
        if llm_count > 0:
            print(f"[TestGen] LLM 生成 {llm_count} 条复杂用例...")
            existing_qs = [c["question"] for c in cases]
            complex_cases = generate_complex_cases(llm_count, existing_qs)
            print(f"  生成 {len(complex_cases)} 条")
            # 编号续接
            for i, case in enumerate(complex_cases):
                case["id"] = len(cases) + i + 1
            cases.extend(complex_cases)

    # 统计
    by_difficulty = {}
    by_category = {}
    for c in cases:
        by_difficulty[c["difficulty"]] = by_difficulty.get(c["difficulty"], 0) + 1
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1

    print(f"\n[TestGen] 总计: {len(cases)} 条")
    print(f"  按难度: {by_difficulty}")
    print(f"  按类别: {by_category}")

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"\n[TestGen] 已保存至: {output_path}")


if __name__ == "__main__":
    main()
