'''
批量问答：从 question_public.csv 读取问题，调用 Agent RAG 回答，输出 submission.csv
'''
import sys
import os
import csv
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from agent.react_agent import ReactAgent
from utils.logger_handler import logger

QUESTION_CSV = r"D:\Code\projects\python\AI\python_ai\question_public.csv"
OUTPUT_CSV = r"D:\Code\projects\python\AI\python_ai\submission6.csv"
TIMEOUT_PER_QUESTION = 120  # 每题最长等待秒数


def clean_question(raw: str) -> str:
    """
    清洗问题文本：
    - 去除多余的引号
    - 合并多行子问题
    - 去除首尾空白
    """
    # 移除包围的引号
    text = raw.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # 替换 "",  "" 这种分隔符为单个空格
    text = re.sub(r'"\s*,\s*"', ' ', text)
    text = re.sub(r'""', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def read_questions(csv_path: str) -> list[tuple[str, str]]:
    """读取问题 CSV，返回 [(id, cleaned_question), ...]"""
    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)  # id,question

        for row in reader:
            if not row:
                continue
            q_id = row[0].strip()
            # 有些问题跨越多行，需要拼接
            q_text = "".join(row[1:]) if len(row) > 1 else ""
            cleaned = clean_question(q_text)
            if q_id and cleaned:
                questions.append((q_id, cleaned))

    return questions


def validate_pic_images(answer: str) -> str:
    """
    校验回答中的 <PIC> 标记与末尾图片名数组是否一致。
    如果 <PIC> 多于图片名，补充未知占位；如果图片名多于 <PIC>，截断多余的图片名。
    :return: 修正后的回答
    """
    pic_count = answer.count("<PIC>")
    if pic_count == 0:
        return answer

    # 提取末尾的图片名数组
    img_match = re.search(r'\n\s*(\[[\s\S]*?\])\s*$', answer)
    if not img_match:
        logger.warning(f"[校验] 回答中有 {pic_count} 个 <PIC>，但末尾缺少图片名数组")
        return answer

    try:
        names = json.loads(img_match.group(1))
        if not isinstance(names, list):
            return answer
    except (json.JSONDecodeError, TypeError):
        return answer

    if pic_count != len(names):
        logger.warning(f"[校验] <PIC> 数({pic_count}) != 图片名数({len(names)})，自动修正")

    # 修正：取两者较小值对齐
    fixed_count = min(pic_count, len(names))
    fixed_names = names[:fixed_count]

    # 重建回答：替换 <PIC> 数量，更新图片名数组
    # 把多余的 <PIC> 替换为普通文本标记
    parts = answer.split("<PIC>")
    if len(parts) > fixed_count + 1:
        # 有多余的 <PIC>，保留前 fixed_count 个，后面的替换为空
        answer = "<PIC>".join(parts[:fixed_count + 1])
    elif len(parts) < fixed_count + 1:
        pass  # 图片名多于 PIC，不影响

    # 更新末尾数组
    answer = re.sub(
        r'\n\s*(\[[\s\S]*?\])\s*$',
        f"\n{json.dumps(fixed_names, ensure_ascii=False)}",
        answer
    )

    return answer


def clean_answer(text: str) -> str:
    """
    清洗回答文本：
    - 保护末尾的图片名数组不被破坏
    - 去除 markdown 标记（**, ###, 等）
    - 将编号列表转为自然语言
    - 压缩多余空白
    """
    # 1. 提取并保护末尾的图片名数组 ["name1", "name2"]
    image_suffix = ""
    img_match = re.search(r'\n\s*(\[[\s\S]*?\])\s*$', text)
    if img_match:
        image_suffix = img_match.group(0)  # 保留原始格式（含换行）
        text = text[:img_match.start()]

    # 2. 清洗正文部分
    text = text.replace("**", "")
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\d+\.\s+', '', text)
    text = re.sub(r'\s*[•\-]\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    # 3. 重新拼接图片名数组
    if image_suffix:
        text += image_suffix

    return text


def load_checkpoint(csv_path: str) -> set:
    """读取已有结果 CSV，返回已完成的 question ID 集合"""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)  # skip header
        except StopIteration:
            return done
        for row in reader:
            if row and row[0].strip():
                done.add(row[0].strip())
    return done


def _execute_agent(agent: ReactAgent, question: str) -> str:
    """执行 agent 并收集完整回答（供超时包裹用）"""
    full_answer = ""
    for chunk in agent.execute_stream(question):
        full_answer += chunk
    return full_answer


def batch_qa():
    logger.info("=" * 50)
    logger.info("开始批量问答（支持断点续跑）...")
    logger.info("=" * 50)

    # 读取已有 checkpoint，初始化输出文件
    done_ids = load_checkpoint(OUTPUT_CSV)
    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    f_out = open(OUTPUT_CSV, "a", encoding="utf-8", newline="")
    writer = csv.writer(f_out)
    if not file_exists:
        writer.writerow(["id", "ret"])
        f_out.flush()

    # 初始化 Agent
    logger.info("初始化 Agent...")
    agent = ReactAgent()
    logger.info("Agent 初始化完成")

    # 读取问题，过滤已完成的
    all_questions = read_questions(QUESTION_CSV)
    remaining = [(q_id, q) for q_id, q in all_questions if q_id not in done_ids]
    total = len(all_questions)
    completed = len(done_ids)

    logger.info(f"全部问题: {total}，已完成: {completed}，剩余: {len(remaining)}\n")

    if not remaining:
        logger.info("所有问题已处理完毕，无需运行。")
        f_out.close()
        return

    start_time = time.time()
    fail_count = 0
    executor = ThreadPoolExecutor(max_workers=1)

    pbar = tqdm(total=total, initial=completed, desc="批量问答", unit="题", ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for idx, (q_id, question) in enumerate(remaining, completed + 1):
        try:
            # 带超时的 agent 调用
            future = executor.submit(_execute_agent, agent, question)
            full_answer = future.result(timeout=TIMEOUT_PER_QUESTION)

            if not full_answer or not full_answer.strip():
                full_answer = "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"

            # 清洗 + 校验
            clean = clean_answer(full_answer)
            clean = validate_pic_images(clean)
            writer.writerow([q_id, clean])
            f_out.flush()

        except FutureTimeout:
            fail_count += 1
            logger.error(f"[超时] 问题 {q_id} 超过 {TIMEOUT_PER_QUESTION}s 无响应，已跳过")
            writer.writerow([q_id, "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"])
            f_out.flush()
            # 取消挂起的 future
            future.cancel()

        except Exception as e:
            fail_count += 1
            logger.error(f"[错误] 问题 {q_id} 处理失败: {str(e)}")
            writer.writerow([q_id, "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"])
            f_out.flush()

        pbar.update(1)
        pbar.set_postfix(fail=fail_count, last_id=q_id)

    executor.shutdown(wait=False)
    pbar.close()
    f_out.close()

    total_time = time.time() - start_time
    final_done = len(load_checkpoint(OUTPUT_CSV))  # 重新统计实际完成的
    logger.info("=" * 50)
    logger.info(f"批量问答完成！")
    logger.info(f"  总问题: {total}，已完成: {final_done}，失败/超时: {fail_count}")
    logger.info(f"  本轮用时: {total_time:.0f}s, 平均: {total_time/len(remaining):.1f}s/条")
    logger.info(f"结果已保存到: {OUTPUT_CSV}")
    logger.info("=" * 50)


if __name__ == "__main__":
    batch_qa()
