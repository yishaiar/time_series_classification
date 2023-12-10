from mdutils.mdutils import MdUtils
from mdutils import Html
import os

def createMdFile(file_name, title,file_address = os.getcwd()):


    file_name = file_address+'/'+ file_name
    try:
        os.remove(file_name +'.md')
    except:
        pass

    mdFile = MdUtils(file_name, title)
    return mdFile

m = createMdFile(file_name = 'README', title = 'Markdown File Example',
                      file_address=os.getcwd())

# -----------------------------






m.new_header(level=1, title='Overviewvv')  # style is set 'atx' format by default.

m.new_paragraph("blabal..")
m.new_paragraph("**IMPORTANT:** blabal..")
m.new_paragraph()  # Add two jump lines
m.new_paragraph("This is an example of text in which has been added color, bold and italics text.",
                     bold_italics_code='bi', color='purple')#bold_italics_code='bic'
m.new_line("This is an example of line break which has been created with ``new_line`` method.")
m.new_line("This is an inline code which contains bold and italics text and it is centered",
                bold_italics_code='cib', align='center')
image_path = "sunset.jpg"
m.new_paragraph(Html.image(path=image_path, size='300x200', align='center'))
m.write("\n\nThe following text has been written with ``write`` method. You can use markdown directives to write: "
             "**bold**, _italics_, ``inline_code``... or ", align='center',color='green')
m.write('  \n')
link = "https://github.com/didix21/mdutils"
text = "mdutils"
m.new_line('  - Italics inline link: ' + m.new_inline_link(link=link, text=text, bold_italics_code='i'))

# ********************************************************************************************************************
# ***************************************************** Markdown *****************************************************
# ********************************************************************************************************************
m.new_header(level=1, title="algorithms")
m.new_header(2, "Create a Table")
table_header = ["Name", "Description", "Value"]
list_of_strings = ["Items", "Descriptions", "Data"]
for x in range(5):
    list_of_strings.extend(["Item " + str(x), "Description Item " + str(x), str(x)])
m.new_line()
m.new_table(columns=3, rows=6, text=list_of_strings, text_align='center')




m.new_header(level=2, title="alg1 (in table of contents)")
m.new_header(level=2, title='alg1 subheader2', add_table_of_contents='n')
m.new_header(level=3, title='alg1 subheader3')

# Atx's levels 1 and 2 are automatically added to the table of contents unless add_table_of_contents='n'
m.new_paragraph("``create_md_file()`` is the last command that has to be called.")
m.insert_code("import Mdutils\n"
                   "\n"
                   "\n"
                   "mdFile = MdUtils(file_name=\'Example_Markdown\',title=\'Markdown File Example\')\n"
                   "mdFile.create_md_file()", 
                   language='python')







# Create a table of contents
m.new_table_of_contents(table_title='Contents', depth=2)
m.create_md_file()