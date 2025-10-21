import PyPDF2

def pdf_to_text(pdf_path, output_txt_path=None):
    """
    Convert a PDF file to plain text.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_txt_path (str, optional): Path to save the output text file. 
                                        If None, prints to console.
    """
    text = ""
    
    try:
        # Open the PDF file in read-binary mode
        with open(pdf_path, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Iterate through each page and extract text
            for page in pdf_reader.pages:
                text += page.extract_text()
                
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
    # Output the text
    if output_txt_path:
        try:
            with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            print(f"Text successfully saved to {output_txt_path}")
        except Exception as e:
            print(f"Error saving text file: {e}")
    else:
        print(text)

if __name__ == "__main__":
    # Example usage
    input_pdf = input("Enter the path to the PDF file: ").strip()
    output_txt = input("Enter the output text file path (optional, press Enter to print to console): ").strip()
    
    if not output_txt:
        output_txt = None
    
    pdf_to_text(input_pdf, output_txt)
