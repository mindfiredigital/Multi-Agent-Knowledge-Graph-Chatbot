"""
Document preprocessing (packaged under knowledge_graph.v1).
"""
import fitz
import os
import json
import base64
from io import BytesIO
import zipfile
import logging
import tempfile
from pathlib import Path
from typing import Optional

from ..config.config import config

from ..utils.logger_config import get_ingestion_logger

logger = get_ingestion_logger()


def load_pdfs_zip(file_path):
    logger.debug(f"Loading PDFs from ZIP file: {file_path}")
    docs = []
    filenames = []
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            pdf_files = [f for f in z.namelist() if f.endswith('.pdf')]
            logger.debug(f"Found {len(pdf_files)} PDF files in ZIP: {pdf_files}")
            for pdf_file in pdf_files:
                logger.debug(f"Processing PDF: {pdf_file}")
                with z.open(pdf_file) as f:
                    doc = fitz.open(f)
                    docs.append(doc)
                    filenames.append(pdf_file)
        logger.debug(f"Successfully loaded {len(docs)} PDF documents from ZIP")
        return docs, filenames
    except Exception as e:
        logger.error(f"Error loading PDFs from ZIP {file_path}: {e}")
        raise


def load_pdf(file_path):
    logger.debug(f"Loading PDF file: {file_path}")
    try:
        doc = fitz.open(file_path)
        original_pages = len(doc)
        logger.debug(f"PDF loaded successfully. Original pages: {original_pages}")
        return doc
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        raise


def extract_images_from_page(page, page_num):
    logger.debug(f"Extracting images from page {page_num}")
    images = []
    image_list = page.get_images()
    logger.debug(f"Found {len(image_list)} images on page {page_num}")
    for img_index, img in enumerate(image_list):
        try:
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            img_rect = None
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 1:
                    img_rect = block["bbox"]
                    break
            if not img_rect:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    img_rect = img_rects[0]
            if pix.n - pix.alpha < 4:
                img_data = pix.tobytes("png")
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                img_data = pix.tobytes("png")
            base64_img = base64.b64encode(img_data).decode()
            images.append({
                'type': 'image',
                'page': page_num,
                'bbox': img_rect,
                'position_y': img_rect[1] if img_rect else 0,
                'base64': base64_img,
                'format': 'png',
                'width': pix.width,
                'height': pix.height,
                'image_index': img_index
            })
            pix = None
        except Exception as e:
            logger.warning(f"Error extracting image {img_index} from page {page_num}: {e}")
            continue
    logger.debug(f"Successfully extracted {len(images)} images from page {page_num}")
    return images


def extract_tables_from_page(page, page_num):
    logger.debug(f"Extracting tables from page {page_num}")
    tables = []
    try:
        tabs = page.find_tables()
        tab_list = list(tabs)
        logger.debug(f"Found {len(tab_list)} tables on page {page_num}")
        for tab_index, tab in enumerate(tab_list):
            table_data = tab.extract()
            table_bbox = tab.bbox
            structured_table = {
                'type': 'table',
                'page': page_num,
                'bbox': table_bbox,
                'position_y': table_bbox[1],
                'table_index': tab_index,
                'rows': len(table_data),
                'cols': len(table_data[0]) if table_data else 0,
                'data': table_data,
                'html': convert_table_to_html(table_data),
                'text': convert_table_to_text(table_data)
            }
            tables.append(structured_table)
    except Exception as e:
        logger.warning(f"Error extracting tables from page {page_num}: {e}")
    logger.debug(f"Successfully extracted {len(tables)} tables from page {page_num}")
    return tables


def convert_table_to_html(table_data):
    if not table_data:
        return ""
    html = "<table border='1'>\n"
    for row_index, row in enumerate(table_data):
        html += "  <tr>\n"
        for cell in row:
            cell_content = str(cell) if cell is not None else ""
            if row_index == 0:
                html += f"    <th>{cell_content}</th>\n"
            else:
                html += f"    <td>{cell_content}</td>\n"
        html += "  </tr>\n"
    html += "</table>"
    return html


def convert_table_to_text(table_data):
    if not table_data:
        return ""
    text_lines = []
    for row in table_data:
        row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
        text_lines.append(row_text)
    return "\n".join(text_lines)


def extract_content_between_headings_enhanced(doc, start_page, start_y, next_heading, all_headings=None):
    logger.debug(f"Extracting content from page {start_page}, y={start_y} to {next_heading['page'] if next_heading else 'end'}")
    content = []
    if next_heading:
        end_page = next_heading['page']
        end_y = next_heading['position_y']
    else:
        end_page = len(doc)
        end_y = float('inf')
    start_page_idx = start_page
    end_page_idx = end_page if next_heading else len(doc)
    heading_positions = set()
    if all_headings:
        for heading in all_headings:
            heading_positions.add((heading['page'], round(heading['position_y'])))
    for page_idx in range(start_page_idx, min(end_page_idx + 1, len(doc))):
        page = doc[page_idx]
        page_num = page_idx
        page_content = []
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        bbox = span["bbox"]
                        y_pos = bbox[1]
                        if not text:
                            continue
                        page_height = page.rect.height
                        if y_pos < page_height * 0.15 or y_pos > page_height * 0.85:
                            continue
                        is_heading = False
                        if all_headings:
                            for heading in all_headings:
                                if (heading['page'] == page_num and
                                    abs(heading['position_y'] - y_pos) < 5 and
                                    text in heading['text']):
                                    is_heading = True
                                    break
                        if not is_heading:
                            page_content.append({
                                'type': 'text',
                                'text': text,
                                'y_pos': y_pos,
                                'font_size': span["size"],
                                'page': page_num,
                                'bbox': bbox,
                                'flags': span["flags"]
                            })
        images = extract_images_from_page(page, page_num)
        for img in images:
            img['y_pos'] = img['position_y']
            page_content.append(img)
        tables = extract_tables_from_page(page, page_num)
        for table in tables:
            table['y_pos'] = table['position_y']
            page_content.append(table)
        filtered_content = []
        for item in page_content:
            include_item = False
            y_pos = item['y_pos']
            if page_num == start_page and (not next_heading or page_num != end_page):
                if y_pos > start_y + 10:
                    include_item = True
            elif page_num == start_page and next_heading and page_num == end_page:
                if y_pos > start_y + 10 and y_pos < end_y - 5:
                    include_item = True
            elif page_num == end_page and next_heading and page_num != start_page:
                if y_pos < end_y - 5:
                    include_item = True
            elif start_page < page_num < end_page:
                include_item = True
            elif not next_heading and page_num >= start_page:
                if y_pos > start_y + 10:
                    include_item = True
            if include_item:
                filtered_content.append(item)
        filtered_content.sort(key=lambda x: x['y_pos'])
        content.extend(filtered_content)
    logger.debug(f"Extracted {len(content)} content items between headings")
    return content


def extract_headings_refined(doc, min_font_size=None, check_bold=True, check_position=True):
    if min_font_size is None:
        min_font_size = config.document_processing.min_font_size
    logger.debug(f"Extracting headings with min_font_size={min_font_size}, check_bold={check_bold}, check_position={check_position}")
    headings = []
    for page_num, page in enumerate(doc):
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]
                        bbox = span["bbox"]
                        flags = span["flags"]
                        if not text:
                            continue
                        is_large_font = font_size >= min_font_size
                        is_bold = bool(flags & 2**4) if check_bold else True
                        is_top_positioned = bbox[1] < page.rect.height * 0.2 if check_position else True
                        is_short = len(text.split()) <= config.document_processing.max_heading_words
                        if is_large_font and (is_bold or is_top_positioned) and is_short:
                            headings.append({
                                'page': page_num,
                                'text': text,
                                'font_size': font_size,
                                'is_bold': is_bold,
                                'position_y': bbox[1],
                                'bbox': bbox
                            })
    logger.debug(f"Found {len(headings)} potential headings across {len(doc)} pages")
    return headings


def merge_compound_headings(headings, doc):
    logger.debug(f"Merging compound headings from {len(headings)} candidates")
    merged = []
    i = 0
    while i < len(headings):
        current = headings[i]
        if i < len(headings) - 1:
            next_heading = headings[i + 1]
            same_page = current['page'] == next_heading['page']
            close_position = abs(current['position_y'] - next_heading['position_y']) < 80
            no_content_between = check_content_between_headings(doc, current, next_heading) if same_page else False
            should_merge = False
            if same_page and close_position and no_content_between:
                similar_font_size = abs(current['font_size'] - next_heading['font_size']) < 2
                is_chapter_pattern = is_chapter_title_pattern(current, next_heading)
                is_title_continuation = is_title_continuation_pattern(current, next_heading)
                should_merge = similar_font_size or is_chapter_pattern or is_title_continuation
            if should_merge:
                merged_text = current['text']
                merged_heading = current.copy()
                max_font_size = max(current['font_size'], next_heading['font_size'])
                merged_heading['font_size'] = max_font_size
                j = i + 1
                while j < len(headings) and headings[j]['page'] == current['page']:
                    candidate = headings[j]
                    close_to_group = abs(candidate['position_y'] - current['position_y']) < 80
                    no_content_to_candidate = check_content_between_headings(doc, current, candidate)
                    fits_pattern = (
                        abs(candidate['font_size'] - current['font_size']) < 2 or
                        abs(candidate['font_size'] - next_heading['font_size']) < 2 or
                        is_chapter_title_pattern(current, candidate) or
                        is_title_continuation_pattern(current, candidate)
                    )
                    if close_to_group and no_content_to_candidate and fits_pattern:
                        merged_text += " " + candidate['text']
                        max_font_size = max(max_font_size, candidate['font_size'])
                        merged_heading['font_size'] = max_font_size
                        j += 1
                    else:
                        break
                merged_heading['text'] = merged_text
                logger.debug(f"Final merged heading: '{merged_text}'")
                merged.append(merged_heading)
                i = j
            else:
                merged.append(current)
                i += 1
        else:
            merged.append(current)
            i += 1
    logger.debug(f"Merged headings: {len(headings)} -> {len(merged)}")
    return merged


def is_chapter_title_pattern(heading1, heading2):
    import re
    text1 = heading1['text'].strip()
    text2 = heading2['text'].strip()
    if re.match(r'^CHAPTER\s+\d+$', text1, re.IGNORECASE):
        return True
    if (heading1['font_size'] < heading2['font_size'] and 
        len(text1.split()) <= 3 and 
        len(text2.split()) >= 3):
        return True
    if (len(text1.split()) <= 2 and 
        len(text2.split()) >= 4 and
        heading2['font_size'] > 20):
        return True
    return False


def is_title_continuation_pattern(heading1, heading2):
    both_large = heading1['font_size'] >= 20 and heading2['font_size'] >= 20
    similar_large_font = (abs(heading1['font_size'] - heading2['font_size']) < 5 and 
                         min(heading1['font_size'], heading2['font_size']) >= config.document_processing.heading_font_threshold_merge)
    return both_large or similar_large_font


def check_content_between_headings(doc, heading1, heading2):
    if heading1['page'] != heading2['page']:
        return False
    page_idx = heading1['page']
    if page_idx < 0 or page_idx >= len(doc):
        return False
    page = doc[page_idx]
    page_height = page.rect.height
    start_y = heading1['position_y'] + 15
    end_y = heading2['position_y'] - 5
    text_blocks = page.get_text("dict")["blocks"]
    content_between = []
    for block in text_blocks:
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    bbox = span["bbox"]
                    y_pos = bbox[1]
                    if not text:
                        continue
                    if y_pos < page_height * 0.15 or y_pos > page_height * 0.85:
                        continue
                    if start_y <= y_pos <= end_y:
                        if len(text.strip()) > 2 and not text.strip().isdigit():
                            content_between.append(text)
    total_content = " ".join(content_between).strip()
    word_count = len(total_content.split()) if total_content else 0
    return word_count < 5


def determine_heading_level(font_size):
    if font_size >= config.document_processing.heading_font_threshold_large:
        return "H1 (Chapter/Main Title)"
    elif font_size >= config.document_processing.heading_font_threshold_medium:
        return "H2 (Major Section)"
    elif font_size >= config.document_processing.heading_font_threshold_small:
        return "H3 (Subsection)"
    else:
        return "H4 (Minor Heading)"


def create_improved_chunks_enhanced(doc, headings, filename=None):
    logger.debug(f"Creating enhanced chunks from {len(headings)} headings")
    sorted_headings = sorted(headings, key=lambda x: (x['page'], x['position_y']))
    merged_headings = merge_compound_headings(sorted_headings, doc)
    chunks = []
    parent_stack = []
    if filename:
        base_filename = os.path.splitext(os.path.basename(filename))[0]
    else:
        base_filename = "Document"
    for i, heading in enumerate(merged_headings):
        heading_level = determine_heading_level(heading['font_size'])
        current_level = int(heading_level[1]) if heading_level.startswith('H') else 1
        parent_id = None
        parent_heading = None
        parent_stack = [entry for entry in parent_stack if entry['level'] < current_level]
        if current_level == 1:
            parent_id = None
            parent_heading = base_filename
        elif parent_stack:
            parent_entry = parent_stack[-1]
            parent_id = parent_entry['id']
            parent_heading = parent_entry['heading']
        else:
            parent_id = None
            parent_heading = base_filename
        chunk = {
            'id': i + 1,
            'heading': heading['text'],
            'heading_level': heading_level,
            'level': current_level,
            'parent_id': parent_id,
            'parent_heading': parent_heading,
            'page': heading['page'],
            'font_size': heading['font_size'],
            'is_bold': heading['is_bold'],
            'content': [],
            'images': [],
            'tables': [],
            'text_content': []
        }
        parent_stack.append({
            'id': chunk['id'],
            'heading': chunk['heading'],
            'level': current_level
        })
        next_heading = merged_headings[i + 1] if i + 1 < len(merged_headings) else None
        content = extract_content_between_headings_enhanced(
            doc,
            heading['page'],
            heading['position_y'],
            next_heading,
            merged_headings
        )
        for item in content:
            if item['type'] == 'text':
                chunk['text_content'].append(item)
            elif item['type'] == 'image':
                chunk['images'].append(item)
            elif item['type'] == 'table':
                chunk['tables'].append(item)
        chunk['content'] = content
        text_content = []
        current_line_texts = []
        last_y_pos = None
        for item in content:
            if item['type'] == 'text':
                current_y_pos = item.get('y_pos', 0)
                if last_y_pos is not None and abs(current_y_pos - last_y_pos) > 3:
                    if current_line_texts:
                        line_text = " ".join(current_line_texts).strip()
                        if line_text:
                            text_content.append(line_text)
                        current_line_texts = []
                current_line_texts.append(item['text'].strip())
                last_y_pos = current_y_pos
        if current_line_texts:
            line_text = " ".join(current_line_texts).strip()
            if line_text:
                text_content.append(line_text)
        chunk['content_text'] = " ".join(text_content)
        chunk['content_text'] = " ".join(chunk['content_text'].split())
        chunk['content_text'] = chunk['content_text'].replace(" .", ".").replace(" ,", ",")
        chunk['content_text'] = chunk['content_text'].replace("- ", "").replace("• ", "")
        chunk['word_count'] = len(chunk['content_text'].split())
        chunk['image_count'] = len(chunk['images'])
        chunk['table_count'] = len(chunk['tables'])
        logger.debug(f"Chunk {i+1} stats: {chunk['word_count']} words, {chunk['image_count']} images, {chunk['table_count']} tables, parent: {parent_heading}")
        chunks.append(chunk)
    logger.debug(f"Created {len(chunks)} enhanced chunks with parent relationships")
    return chunks


def export_enhanced_chunks_to_json(chunks, filename=None):
    logger.debug(f"Preparing {len(chunks)} chunks for export")
    export_data = []
    for chunk in chunks:
        chunk_data = {
            'id': chunk['id'],
            'heading': chunk['heading'],
            'heading_level': chunk['heading_level'],
            'level': chunk['level'],
            'parent_id': chunk['parent_id'],
            'parent_heading': chunk['parent_heading'],
            'page': chunk['page'],
            'font_size': chunk['font_size'],
            'word_count': chunk['word_count'],
            'image_count': chunk['image_count'],
            'table_count': chunk['table_count'],
            'content': chunk['content_text'],
            'images': [
                {
                    'page': img['page'],
                    'bbox': img['bbox'],
                    'base64': img['base64'],
                    'format': img['format'],
                    'width': img['width'],
                    'height': img['height']
                } for img in chunk['images']
            ],
            'tables': [
                {
                    'page': table['page'],
                    'bbox': table['bbox'],
                    'rows': table['rows'],
                    'cols': table['cols'],
                    'data': table['data'],
                    'html': table['html'],
                    'text': table['text']
                } for table in chunk['tables']
            ]
        }
        export_data.append(chunk_data)
    if filename:
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Enhanced chunks exported to {filename}")
            logger.debug(f"Enhanced chunks successfully written to {filename}")
        except Exception as e:
            logger.error(f"Error writing to file {filename}: {e}")
    else:
        logger.debug("No filename provided; skipping export to disk")
    logger.debug("Enhanced chunks successfully converted for downstream use")
    return export_data


def extract_text_from_uploaded_file(file_content: bytes, filename: str, file_type: Optional[str] = None):
    logger.info(f"Processing uploaded file: {filename}")
    if not file_type:
        if filename.lower().endswith('.pdf'):
            file_type = 'application/pdf'
        elif filename.lower().endswith('.txt'):
            file_type = 'text/plain'
        else:
            file_type = 'text/plain'
    result = {
        "text": "",
        "success": False,
        "original_filename": filename,
        "file_type": file_type,
        "error": None
    }
    try:
        if file_type == 'application/pdf' or filename.lower().endswith('.pdf'):
            logger.debug(f"Processing PDF file: {filename}")
            chunks_result = extract_pdf_from_bytes(file_content, filename)
            if isinstance(chunks_result, list):
                logger.debug(f"Successfully extracted {len(chunks_result)} chunks from {filename}")
                return chunks_result
            else:
                logger.error(f"Error extracting from {filename}: {chunks_result.get('error', 'Unknown error')}")
                return []
        else:
            logger.error(f"Unsupported file type for {filename}: {file_type}")
            return []
    except Exception as e:
        logger.error(f"Error processing uploaded file {filename}: {e}")
        return []
    return []


def extract_pdf_from_bytes(file_content: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_content)
        temp_path = Path(temp_file.name)
    try:
        result = apply_preprocessing_single(temp_path, file_name=filename)
        logger.debug(f"Successfully processed PDF {filename} via temporary file")
        return result
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}")
        return {
            "text": "",
            "success": False,
            "error": str(e),
            "original_filename": filename
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()
            logger.debug(f"Temporary file deleted: {temp_path}")


def apply_preprocessing_zip(file_path):
    logger.info(f"Starting preprocessing for ZIP file: {file_path}")
    pdf_docs, filenames = load_pdfs_zip(file_path)
    preprocessed_pdf_docs = []
    for i, (doc, filename) in enumerate(zip(pdf_docs, filenames)):
        logger.debug(f"Processing PDF document {i+1}/{len(pdf_docs)}: {filename}")
        refined_headings = extract_headings_refined(doc, min_font_size=11, check_bold=True, check_position=False)
        enhanced_chunks = create_improved_chunks_enhanced(doc, refined_headings, filename)
        chunks_json = export_enhanced_chunks_to_json(enhanced_chunks)
        preprocessed_pdf_docs.append(chunks_json)
    logger.info(f"Completed preprocessing of {len(preprocessed_pdf_docs)} PDF documents from ZIP")
    return preprocessed_pdf_docs


def apply_preprocessing_single(file_path, file_name):
    logger.info(f"Starting preprocessing for single PDF: {file_path}")
    doc = load_pdf(file_path)
    refined_headings = extract_headings_refined(doc, min_font_size=11, check_bold=True, check_position=False)
    enhanced_chunks = create_improved_chunks_enhanced(doc, refined_headings, file_name)
    chunks_json = export_enhanced_chunks_to_json(enhanced_chunks)
    logger.info("Completed preprocessing for single PDF")
    return chunks_json


