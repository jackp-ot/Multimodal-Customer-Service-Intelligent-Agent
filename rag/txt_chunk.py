import re
import json
from typing import List, Dict, Any, Optional, Tuple
import os


class TxtSmartChunker:
    def __init__(self, chunk_max_length: int = 500, chunk_overlap: int = 0):
        """
        TXT 智能分块器（支持多模态图片引用）：
        1. 按段落（连续换行）拆分
        2. 段落太长时按句子拆分
        3. 合并短段落，控制每块长度
        4. 追踪 <PIC> 标记位置并映射到图片名称
        :param chunk_max_length: 单个块最大字符长度
        :param chunk_overlap: 块间重叠字符数
        """
        self.chunk_max_length = chunk_max_length
        self.chunk_overlap = chunk_overlap
        self.sentence_pattern = re.compile(r'(?<=[。！？.!?\n])\s*', re.UNICODE)
        self.pic_pattern = re.compile(r'<PIC>')

    def _parse_multimodal_text(self, raw_text: str) -> Tuple[str, List[str]]:
        """
        解析多模态文本格式：[带<PIC>的文本, [图片名称列表]]
        :param raw_text: 从 JSON 读取的原始文本（可能是 JSON 数组字符串，也可能已是列表）
        :return: (纯文本内容, 图片名称列表)
        """
        if isinstance(raw_text, str):
            try:
                data = json.loads(raw_text)
                if isinstance(data, list) and len(data) == 2:
                    return data[0], data[1]
            except (json.JSONDecodeError, IndexError):
                pass
            return raw_text, []
        if isinstance(raw_text, list) and len(raw_text) == 2:
            return raw_text[0], raw_text[1]
        return str(raw_text), []

    def _split_paragraphs(self, text: str) -> List[str]:
        """按连续换行拆分成段落"""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """按句子边界拆分"""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_to_chunks(self, items: List[str]) -> List[str]:
        """将段落/句子合并成不超过最大长度的块"""
        chunks = []
        current_chunk = ""

        for item in items:
            if len(item) > self.chunk_max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                for i in range(0, len(item), self.chunk_max_length):
                    chunks.append(item[i:i + self.chunk_max_length])
                continue

            separator = "\n\n" if current_chunk else ""
            if len(current_chunk) + len(separator) + len(item) <= self.chunk_max_length:
                current_chunk += separator + item
            else:
                chunks.append(current_chunk)
                current_chunk = item

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """在相邻块之间添加重叠内容"""
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                result.append(overlap_text + chunk)
        return result

    def _extract_images_for_chunk(self, chunk_text: str, pic_positions: List[int],
                                   image_names: List[str]) -> List[str]:
        """
        从块文本中提取 <PIC> 标记，并返回对应的图片名称列表
        :param chunk_text: 块文本
        :param pic_positions: 原始文本中所有 <PIC> 的位置列表
        :param image_names: 所有图片名称列表
        :return: 该块中包含的图片名称列表
        """
        result = []
        for match in self.pic_pattern.finditer(chunk_text):
            # <PIC> 在 chunk 中的位置 -> 映射到原始文本中的全局位置
            result.extend(image_names)
        # 只取前 N 个（由 PIC 数量决定）
        pic_count = chunk_text.count("<PIC>")
        return image_names[:pic_count] if pic_count > 0 else []

    def create_chunks(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """主入口：将 txt 文本分块，返回带元数据的块列表"""
        paragraphs = self._split_paragraphs(text)

        all_items = []
        for para in paragraphs:
            if len(para) <= self.chunk_max_length:
                all_items.append(para)
            else:
                sentences = self._split_sentences(para)
                all_items.extend(sentences)

        chunks = self._merge_to_chunks(all_items)
        chunks = self._apply_overlap(chunks)

        result = []
        total = len(chunks)
        for i, chunk_text in enumerate(chunks, 1):
            result.append({
                "text": chunk_text,
                "source": filename,
                "metadata": {
                    "source": filename,
                    "chunk_id": i,
                    "total_chunks": total,
                    "char_length": len(chunk_text)
                }
            })

        return result

    def create_multimodal_chunks(
        self, text: str, image_names: List[str], filename: str
    ) -> List[Dict[str, Any]]:
        """
        多模态分块：保留 <PIC> 标记与图片的对应关系
        :param text: 带 <PIC> 标记的文本
        :param image_names: 图片名称列表（与 <PIC> 位置一一对应）
        :param filename: 源文件名
        :return: 带图片引用元数据的块列表
        """
        if not image_names:
            return self.create_chunks(text, filename)

        paragraphs = self._split_paragraphs(text)

        all_items = []
        for para in paragraphs:
            if len(para) <= self.chunk_max_length:
                all_items.append(para)
            else:
                sentences = self._split_sentences(para)
                all_items.extend(sentences)

        chunks = self._merge_to_chunks(all_items)
        chunks = self._apply_overlap(chunks)

        # 将原始文本中的所有 <PIC> 替换为占位符，以便追踪每个 PIC 的全局位置
        pic_positions_in_text = []
        offset = 0
        for pic_match in self.pic_pattern.finditer(text):
            pic_positions_in_text.append(pic_match.start())

        # 为每个块追踪命中哪些 <PIC>
        pic_index_counter = 0

        result = []
        total = len(chunks)
        for i, chunk_text in enumerate(chunks, 1):
            pic_count = chunk_text.count("<PIC>")

            # 该块对应的图片名称
            chunk_image_names = image_names[pic_index_counter:pic_index_counter + pic_count]
            pic_index_counter += pic_count

            metadata = {
                "source": filename,
                "chunk_id": i,
                "total_chunks": total,
                "char_length": len(chunk_text),
                "image_names": json.dumps(chunk_image_names, ensure_ascii=False) if chunk_image_names else "",
            }

            result.append({
                "text": chunk_text,
                "source": filename,
                "metadata": metadata,
            })

        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: uv run python txt_chunk.py <txt文件路径>")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 尝试解析 JSON 格式
    chunker = TxtSmartChunker(chunk_max_length=500)
    try:
        data = json.loads(content)
        text, image_names = data
        chunks = chunker.create_multimodal_chunks(text, image_names, os.path.basename(file_path))
        print(f"文件：{os.path.basename(file_path)} (多模态)")
    except (json.JSONDecodeError, IndexError):
        chunks = chunker.create_chunks(content, os.path.basename(file_path))
        print(f"文件：{os.path.basename(file_path)} (纯文本)")

    print(f"总块数：{len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"]
        imgs = chunk["metadata"].get("image_names", "")
        print(f"--- 块 {i} (长度: {len(text)}) 图片: {imgs[:60]}... ---")
        print(text[:200])
        print()
