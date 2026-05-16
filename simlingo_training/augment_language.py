"""
口语化指令改写 —— 规则改写 + 训练时模板匹配。

流程：
  1. 加载 dreamer.json 模板
  2. 用规则（同义词替换、语序调整、加口语前缀）生成 10 个变体
  3. 占位符 <XXX> 原样保留
  4. 训练时：完整句子匹配回原始模板 → 提取参数 → 随机选变体 → 填入参数

用法：
  python augment_language.py
"""

import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple


# ============================================================
# 规则改写引擎
# ============================================================

# 同义词/口语化替换对：动词、介词短语等
VERB_SYNONYMS = [
    # (原始词/短语, [口语化替换...])
    ("Shift", ["Move over", "Go over", "Slide over", "Switch over", "Change over"]),
    ("Move", ["Shift", "Go", "Head", "Pull"]),
    ("Navigate", ["Head", "Go", "Drive", "Steer"]),
    ("Transition", ["Move over", "Switch", "Change", "Shift over"]),
    ("Adjust", ["Tweak", "Correct", "Fix", "Nudge"]),
    ("Merge", ["Move into", "Join", "Get into", "Slide into"]),
    ("Steer", ["Turn", "Guide", "Point", "Head"]),
    ("Direct", ["Point", "Aim", "Guide", "Head"]),
    ("Guide", ["Direct", "Lead", "Steer", "Point"]),
    ("Accelerate", ["Speed up", "Go faster", "Pick up speed", "Step on it"]),
    ("Decelerate", ["Slow down", "Ease off", "Reduce speed", "Take it easy"]),
    ("Increase", ["Bump up", "Raise", "Turn up", "Push up"]),
    ("Decrease", ["Lower", "Drop", "Bring down", "Ease down"]),
    ("Reduce", ["Lower", "Cut", "Drop", "Bring down"]),
    ("Maintain", ["Keep", "Hold", "Stay at", "Stick to"]),
    ("Approach", ["Head toward", "Move toward", "Go near", "Get close to"]),
    ("Advance", ["Move forward", "Go ahead", "Proceed", "Push forward"]),
    ("Proceed", ["Go", "Continue", "Move on", "Keep going"]),
    ("Crash", ["Hit", "Bump into", "Run into", "Smash into"]),
    ("Stop", ["Come to a stop", "Halt", "Pull up", "Brake"]),
    ("Avoid", ["Steer clear of", "Dodge", "Go around", "Stay away from"]),
    ("Follow", ["Stick to", "Stay on", "Go along", "Keep to"]),
    ("Drive", ["Go", "Head", "Move", "Cruise"]),
    ("Pass", ["Go past", "Drive through", "Cross", "Get through"]),
    ("Reach", ["Get to", "Arrive at", "Hit", "Make it to"]),
    ("Cross", ["Go across", "Pass over", "Drive through", "Traverse"]),
    ("Enter", ["Go into", "Pull into", "Drive into", "Get into"]),
    ("Exit", ["Leave", "Pull out of", "Get out of", "Drive out of"]),
    ("Turn", ["Make a turn", "Take a turn", "Go", "Head"]),
    ("Stay", ["Remain", "Keep", "Hold position", "Wait"]),
]

# 口语化前缀（用于指令型模板）
COMMAND_PREFIXES = [
    "",
    "OK, ",
    "Alright, ",
    "Could you ",
    "Please ",
    "I need you to ",
    "Go ahead and ",
    "Now ",
    "Just ",
]

# 口语化前缀（用于描述型模板，不加 "Could you" 等指令前缀）
DESC_PREFIXES = [
    "",
    "OK, ",
    "Alright, ",
    "Now ",
    "So ",
    "Right now, ",
]

# 口语化后缀
SUFFIXES = [
    "",
    ", please.",
    ", OK?",
    ", alright?",
    ".",
]

# 判断模板是指令还是描述
def _is_command(template: str) -> bool:
    """指令型：以动词开头或含 <PLACEHOLDER>"""
    if '<' in template:
        return True
    first_word = template.strip().split()[0] if template.strip() else ''
    return first_word.lower() not in ('the', 'a', 'an', 'this', 'that', 'it', 'there')

# 整个短语替换模板
PHRASE_SWAPS = [
    # (原始短语模式, [替换...])
    ("in the <LANE_CHANGE_SIDE> direction", [
        "to the <LANE_CHANGE_SIDE>",
        "toward the <LANE_CHANGE_SIDE>",
        "<LANE_CHANGE_SIDE>-side",
    ]),
    ("to the <LANE_CHANGE_SIDE>", [
        "toward the <LANE_CHANGE_SIDE>",
        "in the <LANE_CHANGE_SIDE> direction",
        "<LANE_CHANGE_SIDE>-side",
    ]),
    ("<LANE_OR_LANES> to the <LANE_CHANGE_SIDE>", [
        "<LANE_OR_LANES> toward the <LANE_CHANGE_SIDE>",
        "<LANE_OR_LANES> to the <LANE_CHANGE_SIDE> side",
        "to the <LANE_CHANGE_SIDE> by <LANE_OR_LANES>",
    ]),
    ("in <DISTANCE> meters", [
        "within <DISTANCE> meters",
        "over <DISTANCE> meters",
        "in about <DISTANCE> meters",
    ]),
    ("Drive at <TARGET_SPEED>", [
        "Go at <TARGET_SPEED>",
        "Cruise at <TARGET_SPEED>",
        "Keep it at <TARGET_SPEED>",
        "Maintain <TARGET_SPEED>",
        "Hold your speed at <TARGET_SPEED>",
    ]),
]


def apply_verb_synonyms(template: str) -> List[str]:
    """对模板中的动词做同义词替换，生成多个变体。"""
    variants = [template]
    for word, synonyms in VERB_SYNONYMS:
        new = []
        for v in variants:
            if word in v:
                for syn in synonyms:
                    new.append(v.replace(word, syn))
            else:
                new.append(v)
        variants = list(dict.fromkeys(new))  # 去重保序
    return variants


def apply_prefix_suffix(template: str) -> List[str]:
    """加口语化前缀/后缀（根据模板类型选不同前缀）。"""
    variants = []
    base = template.rstrip('.!?; ')
    is_cmd = _is_command(base)
    prefixes = COMMAND_PREFIXES if is_cmd else DESC_PREFIXES

    for prefix in prefixes:
        if prefix and prefix.endswith(' ') and base[0].isupper():
            body = base[0].lower() + base[1:]
        else:
            body = base
        for suffix in SUFFIXES:
            v = (prefix + body + suffix).strip()
            if v and v[0].islower() and not prefix.endswith(' '):
                v = v[0].upper() + v[1:]
            variants.append(v)
    return list(dict.fromkeys(variants))


def apply_phrase_swaps(template: str) -> List[str]:
    """短语模板替换。"""
    variants = [template]
    for pattern, replacements in PHRASE_SWAPS:
        new = []
        for v in variants:
            if pattern in v:
                for repl in replacements:
                    new.append(v.replace(pattern, repl))
            else:
                new.append(v)
        variants = list(dict.fromkeys(new))
    return variants


def generate_variants(template: str, num_variants: int = 10) -> List[str]:
    """
    用规则为模板生成 N 个口语化变体，始终保留 <PLACEHOLDER>。
    1. 动词同义词替换 → 生成 synonyms 个变体
    2. 短语模板替换 → 进一步扩展
    3. 加前缀后缀 → 口语化表达
    4. 随机采样 num_variants 个
    """
    # 提取占位符
    placeholders = re.findall(r'<[A-Z_]+>', template)

    # Step 1: 动词同义词替换
    v1 = apply_verb_synonyms(template)

    # Step 2: 短语替换
    v2 = []
    for v in v1:
        v2.extend(apply_phrase_swaps(v))
    v2 = list(dict.fromkeys(v2))

    # Step 3: 前缀后缀
    v3 = []
    for v in v2:
        v3.extend(apply_prefix_suffix(v))
    v3 = list(dict.fromkeys(v3))

    # 过滤：保留占位符不变、不要太短或太长
    valid = []
    for v in v3:
        if v == template:
            continue
        if len(v) < 5 or len(v) > 500:
            continue
        # 检查占位符是否完好
        v_ph = re.findall(r'<[A-Z_]+>', v)
        if v_ph == placeholders:
            valid.append(v)

    # 去重 + 采样
    valid = list(dict.fromkeys(valid))
    if len(valid) > num_variants:
        random.seed(hash(template) % 100000)
        valid = random.sample(valid, num_variants)
    elif len(valid) < num_variants:
        # 不够就重复填充
        while len(valid) < num_variants:
            valid.append(template)

    return valid


# ============================================================
# 模板匹配：完整句子 → 原始模板 → 提取参数 → 填入变体
# ============================================================

def _template_to_pattern(template: str) -> Tuple[re.Pattern, List[str]]:
    """将 '<XXX>' 替换为捕获组，返回可匹配任意位置的 pattern。"""
    names = []
    parts = []
    last = 0
    for m in re.finditer(r'<([A-Z_]+)>', template):
        parts.append(re.escape(template[last:m.start()]))
        names.append(m.group(1))
        # 匹配数字（含小数和单位）或 1-3 个单词
        parts.append(r'(\d+\.?\d*(?:\s*m(?:eter)?s?)?(?:\s*km/h)?|\w+(?:\s+\w+){0,2})')
        last = m.end()
    parts.append(re.escape(template[last:]))
    # 前缀匹配：模板不需要匹配整个句子，只需匹配开头
    pattern = re.compile('^' + ''.join(parts), re.IGNORECASE)
    return pattern, names


def match_filled_to_template(filled: str, template: str) -> Optional[Tuple[Dict[str, str], str]]:
    """尝试匹配，返回 ({placeholder: value}, suffix) 或 None。"""
    pattern, names = _template_to_pattern(template)
    m = pattern.search(filled.strip())
    if m:
        values = {names[i]: m.group(i + 1) for i in range(len(names))}
        suffix = filled.strip()[m.end():]  # 模板匹配后的剩余文字
        return values, suffix
    return None


def fill_template(template: str, values: Dict[str, str], filled_suffix: str = "") -> str:
    """填入口语化模板的占位符，保留原句的额外后缀。"""
    result = template
    for key, val in values.items():
        result = result.replace(f'<{key}>', val)
    return result + filled_suffix


# ============================================================
# 训练时使用的 Rewriter
# ============================================================

class ColloquialRewriter:
    """训练时加载口语化模板映射，将完整句子匹配回模板、提取参数、回填。"""

    def __init__(self, map_file: Optional[str] = None, prob: float = 0.5):
        self.prob = prob
        self.colloquial_map: Dict[str, List[str]] = {}
        self._match_cache: Dict[str, Optional[Tuple[str, Dict[str, str], str]]] = {}

        if map_file and os.path.exists(map_file):
            with open(map_file, 'r') as f:
                self.colloquial_map = json.load(f)
            print(f"ColloquialRewriter: loaded {len(self.colloquial_map)} colloquial templates")

        # 加载原始模板
        repo_path = os.getcwd()
        template_path = os.path.join(repo_path, "data/augmented_templates/dreamer.json")
        self.original_templates: Dict[str, List[str]] = {}
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                self.original_templates = json.load(f)
            print(f"ColloquialRewriter: loaded {sum(len(v) for v in self.original_templates.values())} "
                  f"original templates")

    def rewrite(self, original_filled: str) -> str:
        if random.random() > self.prob:
            return original_filled
        if not self.colloquial_map:
            return original_filled

        if original_filled in self._match_cache:
            cached = self._match_cache[original_filled]
            if cached is None:
                return original_filled
            orig_template, values, suffix = cached
            variants = self.colloquial_map.get(orig_template)
            if variants:
                return fill_template(random.choice(variants), values, suffix)
            return original_filled

        for tmpl_list in self.original_templates.values():
            for orig_template in tmpl_list:
                if orig_template not in self.colloquial_map:
                    continue
                result = match_filled_to_template(original_filled, orig_template)
                if result is not None:
                    values, suffix = result
                    self._match_cache[original_filled] = (orig_template, values, suffix)
                    variants = self.colloquial_map[orig_template]
                    return fill_template(random.choice(variants), values, suffix)

        self._match_cache[original_filled] = None
        return original_filled


# ============================================================
# 主入口：生成口语化模板映射
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rule-based colloquial template generation")
    parser.add_argument("--input", type=str, default="data/augmented_templates/dreamer.json")
    parser.add_argument("--output", type=str,
                        default="data/augmented_templates/instruction_colloquial_map.json")
    parser.add_argument("--num-variants", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        return

    with open(args.input, 'r') as f:
        templates = json.load(f)

    all_templates = []
    for items in templates.values():
        all_templates.extend(items)
    unique = list(dict.fromkeys(all_templates))
    print(f"Loaded {len(unique)} unique templates from {len(templates)} categories")

    mapping = {}
    for tmpl in unique:
        variants = generate_variants(tmpl, args.num_variants)
        # 去重
        seen = {tmpl}
        clean = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                clean.append(v)
        mapping[tmpl] = clean if clean else [tmpl]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for v in mapping.values() if len(v) > 1)
    total_v = sum(len(v) for v in mapping.values())
    print(f"Saved: {len(mapping)} templates, {n_ok} have variants, {total_v} total variants")
    print(f"Output: {args.output}")

    # 展示例子
    print("\n" + "=" * 80)
    for orig, variants in list(mapping.items())[:5]:
        print(f"\n  原文: {orig}")
        for v in variants[:5]:
            print(f"    → {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
