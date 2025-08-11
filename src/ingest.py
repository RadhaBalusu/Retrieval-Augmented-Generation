import glob
from pypdf import PdfReader

def load_pdf(path): 
    """This function takes the paths of pdf files as arguments ,
    Using PDFReader we store all the text in files into text variable.
    We store the text of each pdf along with source in the docs list"""
    docs =[]

    for file in glob.glob(path):
        reader= PdfReader(file) 
        text=""

        for page in reader.pages:
            text+=page.extract_text()+ "\n"

        docs.append({"source": file,
                    "text":text})
        
    return(docs)
