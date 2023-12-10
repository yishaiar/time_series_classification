from mdutils.mdutils import MdUtils
from mdutils import Html
import os
def list_files(startpath):
    l = ''
    for root, dirs, files in os.walk(startpath):
        if '__' in root or '.' in root or "log" in root:
            continue
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        l += '{}{}/ \n'.format(indent, os.path.basename(root))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if  'py' not in f and 'md' not in f:
                continue
            l += '{}{}\n'.format(subindent, f)
    return l

def getResults(m,folder,model_name = None, imageList = ['classification_report.png','confusion_matrix.png','results.png']):
    add = f'{folder}/figures/'
    add = add + model_name + '_'  if model_name is not None else add#figure for specific model (when more than single model in folder)

    for image in imageList:
        m.new_paragraph(Html.image(path= add + image, align='center'))
        m.new_paragraph()  # Add two jump lines

    return m


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



# print repo tree structure



m.new_header(level=1, title='Overview')  # style is set 'atx' format by default.

m.new_paragraph("This repo contains experimental code used to implement both classical and "
                "deep learning methods for the task of univariate time series classification of short time series.")
m.new_paragraph("Classification is a supervised learning method used to classify observations into categorical classes "
                "(unlike regression in which the target is always a continuous number). "
                "Time series classification is the task of predicting the class of each time serie according to its temporal features."
                )
m.new_line(m.new_inline_link(link="https://github.com/6110-CTO/classification_yishai" , 
                                                           text="github link" , 
                                                           bold_italics_code='i'))
m.new_line("\n The paper for each of the algorithm is provided in the corresponding section.")
                                                                

m.new_line(' - ' +"The code is implementation libraries are: pytorch-lightning, sktime, sklearn, xgboost and clearml ")

m.new_line(' - ' +"todo: fix - IMBALANCED DATA: The data is imbalanced, and the code is not handling it yet.")
m.new_line(' - ' +"todo: fix - The preprocessing and normalization is done using shap, confusion matrix and and accuracy")
m.new_line(' - ' +"todo: fix - The analysis is done using shap, confusion matrix and and accuracy")

m.new_header(level=2, title="Structure of Repo")
m.insert_code(list_files(os.getcwd()))

m.new_line(' - ' + "the code is tested on the UCR time series classification archive " + m.new_inline_link(link="http://www.timeseriesclassification.com/index.php" , 
                                                           text="(link)" , 
                                                           bold_italics_code='i'))


# ********************************************************************************************************************
# ***************************************************** Markdown *****************************************************
# ********************************************************************************************************************
m.new_header(level=1, title="algorithms")




# ********************************************************************************************************************
m.new_header(level=2, title="Dealing with Imbalanced Data")
m.new_paragraph("The code is based on the following papers: todo - add papers and finish implementation")
m.new_header(level=2, title="FC (Fully Connected) Network")

m.new_paragraph("The fully Connected network is a 3 layer implemention using by pytorch lightning."
                "The network is trained with Adam optimizer and CrossEntropyLoss loss (logits output)"
                "The output is the argmax of the network's logits")
m.new_paragraph("The code is based on the following papers: todo - add papers")
m = getResults(m,folder = 'FC')
# ********************************************************************************************************************
m.new_header(level=2, title="Rocket data transformation")
m.new_paragraph("The Rocket data transformation is with the MiniRocketMultivariate using ridge regression classifier.")
m.new_paragraph("The code is based on the following papers: todo - add papers")
m = getResults(m,folder = 'sktime',model_name = 'MiniRocketMultivariate')
# ********************************************************************************************************************
# m.new_header(level=2, title="Transformers Network (Tarnet implementation)")
# m.new_header(level=2, title="XGBoost")
# m.new_header(level=2, title="MLP (Multi Layer Perceptron)")
# m.new_header(level=2, title="CNN (Convolutional Neural Network)")
# m.new_header(level=2, title="LSTM (Long Short Term Memory)")
# m.new_header(level=2, title="CNN-LSTM")
# m.new_header(level=2, title="ResNet")
# ********************************************************************************************************************
m.new_header(level=2, title="InceptionTime")
m.new_paragraph("The InceptionTime: todo ")
m.new_paragraph("The code is based on the following papers: todo - add papers")
m = getResults(m,folder = 'sktime',model_name = 'InceptionTime')
# ********************************************************************************************************************
# m.new_header(level=2, title="TSFEL (Time Series Feature Extraction Library)")
# m.new_header(level=2, title="TSFresh (Time Series Feature Extraction Library)")


"The code is based on the following papers:"
m.new_header(level=2, title='alg1 subheader2', add_table_of_contents='n')
m.new_header(level=3, title='alg1 subheader3')








# Create a table of contents
m.new_table_of_contents(table_title='Contents', depth=2)
m.create_md_file()