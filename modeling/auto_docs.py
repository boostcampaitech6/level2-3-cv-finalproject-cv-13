import os

from docx import Document                       
from docx2pdf import convert
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE       
from docx.shared import Pt, Inches, RGBColor  


document = Document("base_document.docx")

# personal info title
p_info_title = document.add_paragraph("환자 정보")
run = p_info_title.runs[0]
run.font.size = Pt(14)
run.font.bold = True
run.font.name = "맑은 고딕"
run.font.color.rgb = RGBColor(0, 0, 0)

# personal info table
p_info_table = document.add_table(rows=2, cols=5)
p_info_table.style = "custom_style"
p_info_table.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_info = {
    0: ["환자 ID", "이름", "성별", "나이", "생년월일"],
    1: ["P0001", "홍길동", "남", "25", "1996-01-01"]
}

table = document.tables[-1]
for i, row in enumerate(table.rows):
    for j, cell in enumerate(row.cells):       
        # set text and alignment
        for para in cell.paragraphs:
            if i == 0:
                para.add_run(p_info[i][j]).bold = True 
            else:
                para.add_run(p_info[i][j])
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            # set font            
            for run in para.runs:
                run.font.name = "맑은 고딕"
                run.font.size = Pt(12)

document.add_paragraph()

# result title
img_path = os.path.join("../", "backend", "sampleimages", "axial_0000.png")
result_title = document.add_paragraph("검사 결과")
run = result_title.runs[0]
run.font.size = Pt(14)
run.font.bold = True
run.font.name = "맑은 고딕"
run.font.color.rgb = RGBColor(0, 0, 0)

# plane table
plane_table = document.add_table(rows=1, cols=3)
plane_table.alignment = WD_ALIGN_PARAGRAPH.CENTER
plane_table.style = "custom_style2"
plane_info = {
    0: ["Axial", "Coronal", "Sagittal"],
}

table = document.tables[-1]
for i, row in enumerate(table.rows):
    for j, cell in enumerate(row.cells):       
        # set text and alignment
        for para in cell.paragraphs:
            para.add_run(plane_info[i][j]).bold = True 
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            # set font            
            for run in para.runs:
                run.font.name = "맑은 고딕"
                run.font.size = Pt(12)

# image table
img_paragraph = document.add_paragraph()
img_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
img_paths = [img_path, img_path, img_path]
for img_path in img_paths:
    run = img_paragraph.add_run()
    run.add_picture(img_path, width=Inches(2.5))
    img_paragraph.add_run().add_text(" "*3)

document.add_paragraph()

# diseases table
result_table = document.add_table(rows=4, cols=3)
result_table.style = "custom_style"
result_table.alignment = WD_ALIGN_PARAGRAPH.CENTER
result_info = {
    0: ["Normal (정상)", "50%", "O"],
    1: ["Abnormal (비정상)", "50%", ""],
    2: ["ACL (전방십자인대 찢어짐)", "20%", ""],
    3: ["Meniscus (반달연골 찢어짐)", "20%", ""]
}

table = document.tables[-1]
first_col = table.columns[0]
first_col.width = Inches(8)

for i, row in enumerate(table.rows):
    for j, cell in enumerate(row.cells):
        for para in cell.paragraphs:
            para.add_run(result_info[i][j])
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.font.name = "맑은 고딕"
                run.font.size = Pt(12)

document.add_paragraph()


if os.path.exists('modeling_output.docx'):
    os.remove('modeling_output.docx')
document.save('modeling_output.docx')
convert('modeling_output.docx', 'modeling_output.pdf')
os.remove('modeling_output.docx')
