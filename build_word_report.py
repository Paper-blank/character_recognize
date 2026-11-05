"""Build the verified Word version of the KMNIST experiment report."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


TASK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = TASK_DIR / "实验报告.docx"

BODY_FONT = "Noto Serif CJK SC"
HEADING_FONT = "Noto Sans CJK SC"
MONO_FONT = "Noto Sans Mono CJK SC"
TABLE_HEADER = "1F4E78"
TABLE_ALT = "EAF2F8"
TABLE_BORDER = "D9D9D9"
TEXT_GRAY = RGBColor(89, 89, 89)


def set_run_font(
    run,
    font_name: str,
    size: float | None = None,
    bold: bool | None = None,
    color: RGBColor | None = None,
) -> None:
    run.font.name = font_name
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn("w:ascii"), font_name)
    fonts.set(qn("w:hAnsi"), font_name)
    fonts.set(qn("w:eastAsia"), font_name)
    fonts.set(qn("w:cs"), font_name)


def set_style_font(style, font_name: str, size: float, bold: bool = False) -> None:
    style.font.name = font_name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style._element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn("w:ascii"), font_name)
    fonts.set(qn("w:hAnsi"), font_name)
    fonts.set(qn("w:eastAsia"), font_name)
    fonts.set(qn("w:cs"), font_name)


def configure_styles(document: Document) -> None:
    styles = document.styles
    normal = styles["Normal"]
    set_style_font(normal, BODY_FONT, 11)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(6)

    title = styles["Title"]
    set_style_font(title, HEADING_FONT, 24, bold=True)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(84)
    title.paragraph_format.space_after = Pt(18)
    title_properties = title._element.get_or_add_pPr()
    title_borders = title_properties.find(qn("w:pBdr"))
    if title_borders is not None:
        title_properties.remove(title_borders)

    heading_1 = styles["Heading 1"]
    set_style_font(heading_1, HEADING_FONT, 16, bold=True)
    heading_1.paragraph_format.space_before = Pt(16)
    heading_1.paragraph_format.space_after = Pt(8)
    heading_1.paragraph_format.keep_with_next = True

    heading_2 = styles["Heading 2"]
    set_style_font(heading_2, HEADING_FONT, 13, bold=True)
    heading_2.paragraph_format.space_before = Pt(12)
    heading_2.paragraph_format.space_after = Pt(6)
    heading_2.paragraph_format.keep_with_next = True

    caption = styles["Caption"]
    set_style_font(caption, HEADING_FONT, 9.5)
    caption.font.italic = False
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)

    if "Code Block" not in styles:
        code_style = styles.add_style("Code Block", 1)
    else:
        code_style = styles["Code Block"]
    set_style_font(code_style, MONO_FONT, 9.5)
    code_style.paragraph_format.left_indent = Inches(0.25)
    code_style.paragraph_format.right_indent = Inches(0.15)
    code_style.paragraph_format.space_before = Pt(4)
    code_style.paragraph_format.space_after = Pt(4)
    code_style.paragraph_format.line_spacing = 1.15


def set_page_layout(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.78)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.35)
    section.different_first_page_header_footer = True


def add_page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("第 ")
    set_run_font(run, HEADING_FONT, 9, color=TEXT_GRAY)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, text, end])
    suffix = paragraph.add_run(" 页")
    set_run_font(suffix, HEADING_FONT, 9, color=TEXT_GRAY)


def configure_footer(document: Document) -> None:
    footer = document.sections[0].footer
    paragraph = footer.paragraphs[0]
    add_page_field(paragraph)
    first_footer = document.sections[0].first_page_footer
    first_footer.paragraphs[0].text = ""


def add_hyperlink(paragraph, text: str, url: str) -> None:
    part = paragraph.part
    relationship_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    fonts = OxmlElement("w:rFonts")
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), BODY_FONT)
    properties.extend([fonts, color, underline])
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.extend([properties, text_node])
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_body(document: Document, text: str, first_line: bool = True):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Inches(0.29) if first_line else None
    paragraph.paragraph_format.widow_control = True
    run = paragraph.add_run(text)
    set_run_font(run, BODY_FONT, 11)
    return paragraph


def add_rich_body(document: Document, parts: list[tuple[str, bool]], first_line: bool = True):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Inches(0.29) if first_line else None
    paragraph.paragraph_format.widow_control = True
    for text, bold in parts:
        run = paragraph.add_run(text)
        set_run_font(run, BODY_FONT, 11, bold=bold)
    return paragraph


def add_caption(document: Document, text: str, keep_with_next: bool = True):
    paragraph = document.add_paragraph(text, style="Caption")
    paragraph.paragraph_format.keep_with_next = keep_with_next
    return paragraph


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 90, start: int = 95, bottom: int = 90, end: int = 95) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        element = margins.find(qn(f"w:{side}"))
        if element is None:
            element = OxmlElement(f"w:{side}")
            margins.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    properties = table._tbl.tblPr
    borders = properties.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), TABLE_BORDER)
        borders.append(element)


def repeat_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    properties.append(header)


def prevent_row_split(row) -> None:
    properties = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    properties.append(cant_split)


def format_table(
    table,
    widths: list[float],
    narrative_columns: set[int] | None = None,
    font_size: float = 9.5,
) -> None:
    narrative_columns = narrative_columns or set()
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    repeat_header(table.rows[0])
    for row_index, row in enumerate(table.rows):
        prevent_row_split(row)
        for column_index, cell in enumerate(row.cells):
            cell.width = Inches(widths[column_index])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            if row_index == 0:
                set_cell_shading(cell, TABLE_HEADER)
            elif row_index % 2 == 0:
                set_cell_shading(cell, TABLE_ALT)
            for paragraph in cell.paragraphs:
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.LEFT
                    if column_index in narrative_columns and row_index > 0
                    else WD_ALIGN_PARAGRAPH.CENTER
                )
                paragraph.paragraph_format.space_before = Pt(1)
                paragraph.paragraph_format.space_after = Pt(1)
                paragraph.paragraph_format.line_spacing = 1.1
                for run in paragraph.runs:
                    set_run_font(
                        run,
                        HEADING_FONT if row_index == 0 else BODY_FONT,
                        font_size,
                        bold=row_index == 0,
                        color=RGBColor(255, 255, 255) if row_index == 0 else RGBColor(0, 0, 0),
                    )


def add_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    narrative_columns: set[int] | None = None,
    font_size: float = 9.5,
):
    table = document.add_table(rows=1, cols=len(headers))
    for index, value in enumerate(headers):
        table.rows[0].cells[index].text = value
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = value
    format_table(table, widths, narrative_columns, font_size)
    after = document.add_paragraph()
    after.paragraph_format.space_after = Pt(2)
    return table


def set_image_alt_text(run, description: str) -> None:
    drawings = run._element.xpath(".//wp:docPr")
    if drawings:
        drawings[0].set("descr", description)


def add_figure(document: Document, image_path: Path, caption: str, width: float, alt: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(width))
    set_image_alt_text(run, alt)
    add_caption(document, caption, keep_with_next=False)


def shade_paragraph(paragraph, fill: str = "F2F2F2") -> None:
    properties = paragraph._p.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def add_code_block(document: Document, lines: list[str]) -> None:
    for index, line in enumerate(lines):
        paragraph = document.add_paragraph(style="Code Block")
        shade_paragraph(paragraph)
        paragraph.paragraph_format.keep_with_next = index < len(lines) - 1
        run = paragraph.add_run(line if line else " ")
        set_run_font(run, MONO_FONT, 9.5)


def add_cover(document: Document) -> None:
    title = document.add_paragraph("基于 KMNIST 的手写字符识别实验报告", style="Title")
    title.paragraph_format.keep_with_next = True
    title_properties = title._p.get_or_add_pPr()
    title_borders = title_properties.find(qn("w:pBdr"))
    if title_borders is not None:
        title_properties.remove(title_borders)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(42)
    run = subtitle.add_run("课程实验报告")
    set_run_font(run, HEADING_FONT, 14, bold=False, color=TEXT_GRAY)

    summary = document.add_paragraph()
    summary.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    summary.paragraph_format.left_indent = Inches(0.7)
    summary.paragraph_format.right_indent = Inches(0.7)
    summary.paragraph_format.line_spacing = 1.5
    run = summary.add_run(
        "本报告记录了 KMNIST 手写单字符识别任务的数据处理、模型搭建和实验结果。"
        "实验比较了全连接基线模型与改进卷积神经网络，改进模型在官方测试集上的准确率达到 96.72%，"
        "Macro-F1 达到 96.71%。"
    )
    set_run_font(run, BODY_FONT, 11.5)

    document.add_paragraph()
    for text in (
        "数据集  KMNIST",
        "任务类型  手写单字符图像分类",
        "最终测试准确率  96.72%",
        "最终 Macro-F1  96.71%",
    ):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(7)
        run = paragraph.add_run(text)
        set_run_font(run, HEADING_FONT, 11, bold=True)
    document.add_page_break()


def build_report() -> None:
    document = Document()
    configure_styles(document)
    set_page_layout(document)
    configure_footer(document)
    properties = document.core_properties
    properties.title = "基于 KMNIST 的手写字符识别实验报告"
    properties.subject = "KMNIST 手写字符识别课程实验"
    properties.author = ""
    properties.keywords = "KMNIST, OCR, CNN, 手写字符识别"

    add_cover(document)

    document.add_heading("1 GitHub 链接", level=1)
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(8)
    lead = paragraph.add_run("项目代码地址：")
    set_run_font(lead, BODY_FONT, 11)
    placeholder = paragraph.add_run("【待补充 GitHub 链接】")
    set_run_font(placeholder, BODY_FONT, 11, bold=True)
    add_body(
        document,
        "后续把 task5 文件夹上传到自己的仓库后，将上面的占位内容替换成真实链接即可。",
    )

    document.add_heading("2 任务描述", level=1)
    add_body(
        document,
        "本次实验完成的是一个手写单字符识别任务，也可以看作 OCR 中比较基础的一步。模型的输入是一张 28×28 的灰度图片，输出是图片最可能属于的字符类别。这里识别的是 KMNIST 中的 10 个日文平假名草书字符，不涉及一整行文字的检测、字符切分和语义分析。",
    )
    add_body(
        document,
        "为了看出模型改进前后的差别，我做了两组实验。第一组使用简单的全连接神经网络作为基线，先把二维图片拉平成一维向量，再进行分类。第二组改成卷积神经网络，并加入 BatchNorm、Dropout、轻量数据增强和学习率调整。两组实验使用相同的数据划分、训练轮数和随机种子，最后在官方测试集上比较准确率与 Macro-F1。",
    )
    add_body(
        document,
        "我希望通过这次实验回答两个问题：一是简单的神经网络能不能完成基本的字符识别；二是在保留笔画空间关系以后，卷积网络能带来多大的提升。",
    )

    document.add_heading("3 数据描述", level=1)
    document.add_heading("3.1 数据集选择", level=2)
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Inches(0.29)
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = paragraph.add_run("实验使用 ")
    set_run_font(run, BODY_FONT, 11)
    add_hyperlink(paragraph, "KMNIST（Kuzushiji-MNIST）", "https://codh.rois.ac.jp/kmnist/")
    run = paragraph.add_run(
        "。它来自日本古籍中的草书字符，格式和 MNIST 接近，但识别对象是真正的文字字符，比只识别 0～9 数字更符合这次文字图像识别的主题。"
    )
    set_run_font(run, BODY_FONT, 11)
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Inches(0.29)
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = paragraph.add_run("数据集的")
    set_run_font(run, BODY_FONT, 11)
    add_hyperlink(paragraph, "官方代码仓库", "https://github.com/rois-codh/kmnist")
    run = paragraph.add_run("提供了类别映射、基准结果和许可说明，本文通过 ")
    set_run_font(run, BODY_FONT, 11)
    add_hyperlink(
        paragraph,
        "Torchvision 的 KMNIST 接口",
        "https://docs.pytorch.org/vision/main/generated/torchvision.datasets.KMNIST.html",
    )
    run = paragraph.add_run("下载和读取数据。")
    set_run_font(run, BODY_FONT, 11)
    add_body(
        document,
        "数据一共有 70,000 张 28×28 单通道灰度图，分成 10 个类别。其中训练集 60,000 张，每类 6,000 张；测试集 10,000 张，每类 1,000 张。数据集使用 CC BY-SA 4.0 许可，项目中只保存下载脚本和来源说明，不把原始数据提交到 GitHub。",
    )
    add_caption(document, "表 1  KMNIST 类别对应关系")
    add_table(
        document,
        ["标签", "字符", "罗马音", "标签", "字符", "罗马音"],
        [
            ["0", "お", "o", "5", "は", "ha"],
            ["1", "き", "ki", "6", "ま", "ma"],
            ["2", "す", "su", "7", "や", "ya"],
            ["3", "つ", "tsu", "8", "れ", "re"],
            ["4", "な", "na", "9", "を", "wo"],
        ],
        [0.7, 0.8, 1.55, 0.7, 0.8, 1.55],
    )

    document.add_heading("3.2 数据划分和预处理", level=2)
    add_body(
        document,
        "我从官方训练集中按类别均衡抽出 6,000 张作为验证集，剩下 54,000 张用于训练，官方的 10,000 张测试图只用于最后评价。划分时固定随机种子为 2026，因此两组实验拿到的是同一批训练图和验证图，比较会更公平。",
    )
    add_body(
        document,
        "所有图片先转成张量，再使用官方 60,000 张训练图（不包含测试集）的统计量进行归一化，均值为 0.1918，标准差为 0.3483。基线模型不做数据增强。改进模型在训练阶段加入不超过 8° 的随机旋转、8% 的平移和 0.92～1.08 倍缩放，让模型适应一些书写位置和大小变化。增强范围没有设得太大，因为字符的方向和笔画结构本身就是分类依据，过强变换反而可能改变字符特征。",
    )

    document.add_heading("4 代码修改描述", level=1)
    add_body(
        document,
        "这次没有拉取一个完整的开源识别项目，而是在原工程中新建了独立的 task5 目录。这样不会影响前面的植物分类、眼底分割等任务，数据、代码和实验结果也更容易单独整理。",
    )
    add_body(document, "主要新增内容如下：", first_line=False)
    add_caption(document, "表 2  新增文件与主要功能")
    add_table(
        document,
        ["文件", "主要内容"],
        [
            ["models.py", "实现基线 MLP 和改进 CNN，并提供统一的模型创建函数。"],
            ["data_utils.py", "下载 KMNIST，完成归一化、轻量增强、按类别均衡划分，以及外部图片预处理。"],
            ["experiment_utils.py", "封装每轮训练、测试指标、Macro-F1、混淆矩阵、训练曲线和预测样例绘制。"],
            ["train.py", "完成训练与验证，根据验证准确率保存最佳模型，最后在官方测试集上评价。"],
            ["evaluate.py", "读取保存的权重，独立复算测试集指标并重新生成混淆矩阵。"],
            ["predict.py", "支持输入一张自己的字符图片，输出概率最高的若干类别。"],
            ["compare_results.py", "自动读取两组 metrics.json，计算提升幅度并生成结果柱状图。"],
            ["tests/", "增加 9 个单元测试，检查模型输出、类别映射、数据划分、指标口径和图片预处理。"],
            [".gitignore", "忽略原始数据、缓存和模型权重，避免把较大的可再生文件上传到仓库。"],
        ],
        [1.65, 4.75],
        narrative_columns={1},
        font_size=9.2,
    )
    add_body(
        document,
        "基线模型的结构是 784 → 128 → 10，使用 ReLU 激活和 Adam 优化器。这个模型很容易理解，参数也少，但图片在进入网络时被直接拉平了，原来相邻的笔画关系没有被明确利用。",
    )
    add_body(
        document,
        "改进模型使用两组卷积模块，通道数依次为 32 和 64。每个模块先连续执行两次 Conv-BatchNorm-ReLU，再进行最大池化和 Dropout，最后接 128 维全连接层输出 10 个类别。训练时使用 AdamW、余弦学习率衰减和 0.05 的标签平滑。卷积层可以从局部区域提取横、竖、弯钩等笔画信息，BatchNorm 让训练过程更稳定，Dropout 和数据增强主要用于减轻过拟合。",
    )

    document.add_heading("5 实验设置", level=1)
    add_body(
        document,
        "两组实验都训练 10 轮，batch size 为 256，初始学习率为 0.001，随机种子为 2026。损失函数使用交叉熵，模型按照验证集准确率保存最佳权重。运行环境如下：",
    )
    for item in (
        "Python 3.14.6",
        "PyTorch 2.14.0 与 CUDA 13.0",
        "NVIDIA GeForce RTX 4090 24 GB",
        "训练集 54,000 张，验证集 6,000 张，测试集 10,000 张",
    ):
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.left_indent = Inches(0.45)
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(item)
        set_run_font(run, BODY_FONT, 11)
    add_body(
        document,
        "训练时间只反映本次机器上的运行情况，不同显卡或使用 CPU 时会有明显差别。",
    )

    document.add_heading("6 结果对比", level=1)
    document.add_heading("6.1 总体结果", level=2)
    add_caption(document, "表 3  基线模型与改进模型结果对比")
    add_table(
        document,
        ["模型", "参数量", "最佳\n轮次", "最佳验证\n准确率", "测试\n准确率", "测试\nMacro-F1", "训练\n时间"],
        [
            ["基线 MLP", "101,770", "10", "95.40%", "88.92%", "88.92%", "12.60 秒"],
            ["改进 CNN", "468,010", "9", "99.08%", "96.72%", "96.71%", "17.35 秒"],
        ],
        [1.0, 0.85, 0.65, 1.0, 0.85, 1.0, 0.9],
        font_size=8.8,
    )
    add_figure(
        document,
        TASK_DIR / "results" / "model_comparison.png",
        "图 1  两种模型在 KMNIST 测试集上的结果对比",
        6.1,
        "基线 MLP 与改进 CNN 的测试准确率和 Macro-F1 柱状图",
    )
    add_rich_body(
        document,
        [
            ("从结果来看，改进 CNN 的测试准确率从 88.92% 提升到 96.72%，增加了 ", False),
            ("7.80 个百分点", True),
            ("；Macro-F1 增加了 ", False),
            ("7.79 个百分点", True),
            ("。换成样本数来理解，10,000 张测试图中，正确数量由 8,892 张增加到 9,672 张，多识别正确 780 张。测试错误率从 11.08% 降到 3.28%，相对减少约 ", False),
            ("70.40%", True),
            ("。", False),
        ],
    )
    add_body(
        document,
        "这次提升的代价也比较直观。改进模型的参数量约为基线的 4.60 倍，训练时间多了约 4.76 秒。不过整个模型仍然只有 468,010 个可训练参数，在当前显卡上训练 10 轮不到 20 秒。对这个课程实验来说，我认为增加的计算量是可以接受的。",
    )

    document.add_heading("6.2 训练过程", level=2)
    add_body(
        document,
        "基线模型后几轮的训练准确率继续上升，最终达到 99% 左右，但验证准确率停在 95% 左右，训练损失与验证损失之间也出现了差距，说明模型存在一定过拟合。改进模型加入数据增强后训练样本变得更难，训练准确率反而没有基线高，但验证准确率达到了 99.08%。这说明训练集准确率高并不等于泛化能力强，也说明包含 CNN、数据增强和正则化在内的整体改进方案是有效的。由于本次没有另外做消融实验，暂时不能把提升单独归因于其中某一项改动。",
    )
    add_figure(
        document,
        TASK_DIR / "results" / "baseline" / "training_curves.png",
        "图 2  基线 MLP 的训练与验证曲线",
        6.15,
        "基线 MLP 的训练损失、验证损失、训练准确率与验证准确率曲线",
    )
    add_figure(
        document,
        TASK_DIR / "results" / "improved" / "training_curves.png",
        "图 3  改进 CNN 的训练与验证曲线",
        6.15,
        "改进 CNN 的训练损失、验证损失、训练准确率与验证准确率曲线",
    )
    add_body(
        document,
        "随机划分的验证集成绩高于官方测试集成绩，尤其是基线模型差距更明显。一种可能的原因是验证图与训练图来自同一训练划分，样本分布更接近；官方测试集相对更独立，因此更能反映模型对新样本的识别能力。最终比较以官方测试集成绩为准。",
    )

    document.add_heading("6.3 分类情况和错误分析", level=2)
    add_body(
        document,
        "改进模型在 10 类上的 F1 都超过了 93%。其中 ya、re、wo 的 F1 大约为 98.45%、98.07% 和 98.50%，识别比较稳定；su 的 F1 为 93.20%，是 10 类中最低的一类。混淆矩阵显示，最常见的错误是把 su 识别成 tsu，一共有 63 张；其次是把 ki 识别成 ma，一共有 35 张。这些字符在草书写法中有相似的弯曲笔画，图片又只有 28×28，部分细节比较难区分。",
    )
    add_figure(
        document,
        TASK_DIR / "results" / "improved" / "confusion_matrix.png",
        "图 4  改进 CNN 的归一化混淆矩阵",
        5.35,
        "改进 CNN 在 KMNIST 十个类别上的归一化混淆矩阵",
    )
    add_figure(
        document,
        TASK_DIR / "results" / "improved" / "sample_predictions.png",
        "图 5  改进 CNN 的测试集预测样例",
        4.9,
        "改进 CNN 的八个错误预测和八个正确预测样例",
    )

    document.add_heading("7 总结与不足", level=1)
    add_body(
        document,
        "本次实验完成了从数据下载、模型训练到独立评估和单图预测的一套字符识别流程。结果说明，全连接网络已经能识别大部分字符，但拉平操作不便于模型直接利用二维邻域和局部笔画结构；换成 CNN 并加入适当的数据增强与正则化后，测试准确率提高到了 96.72%，错误率也明显下降。",
    )
    add_body(
        document,
        "目前的任务仍然有几个限制。第一，模型只能在已经裁剪好的单字符图片上工作，还不能直接处理包含多个字符的照片。第二，类别只有 KMNIST 提供的 10 类，不能识别常用汉字、英文字母或数字。第三，单图预测中的居中和缩放规则比较简单，如果输入图片背景复杂或字符没有分割好，结果可能下降。后续如果继续改进，可以增加字符检测与切分步骤，扩展到更多类别，也可以专门分析 su、tsu 等易混类别并尝试注意力机制。",
    )

    document.add_heading("8 复现方法", level=1)
    add_body(document, "在项目根目录依次执行以下命令：", first_line=False)
    add_code_block(
        document,
        [
            "python -m pip install -r task5/requirements.txt",
            "",
            "python task5/train.py --model baseline --epochs 10 --seed 2026",
            "python task5/train.py --model improved --epochs 10 --seed 2026",
            "python task5/compare_results.py",
            "",
            "pytest -q task5/tests",
        ],
    )
    final_paragraph = add_body(
        document,
        "详细数值保存在 task5/results/baseline/metrics.json、task5/results/improved/metrics.json 和 task5/results/comparison.json 中。原始数据会在第一次训练时自动下载，因此不用手动准备数据文件。",
    )
    final_paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    document.save(OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    build_report()
