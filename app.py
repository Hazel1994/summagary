import os

temp_dir = './temp/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TMPDIR'] = temp_dir
import gradio as gr
import shutil
from summagery_pipline import  Summagery

if os.path.exists(temp_dir):
    try:
        shutil.rmtree(temp_dir)
        print(f"The directory at {temp_dir} has been removed successfully along with its contents.")
    except OSError as e:
        print(f"Error: {temp_dir} - {e}")

os.makedirs(temp_dir, exist_ok=True)

def generate(text, batch_size, model_type, abstractness):

    model = Summagery(model_type,batch_size=int(batch_size),abstractness=abstractness)
    images=model.ignite(text)

    return images


with gr.Blocks(theme=gr.themes.Soft(),) as demo:
    gr.Markdown(
        """
        <h1 style="text-align:center;">Welcome to Summagery: Document Summarization through Images</h1>

        <h3 style="text-align:center;">Summarize long and short documents on any topic as images</h3>

        <p style="text-align:left;">1. <b>Document:</b> Enter the text of the document you want to summarize.</p>
        <p style="text-align:left;">2. <b>Batch Size:</b> Adjust the batch size for processing very long documents (e.g., 500 pages)</p>
        <p style="text-align:left;">3. <b>T5_Model_Checkpoint:</b> Choose the model checkpoint (e.g., "t5-large", "t5-base", "t5-small"). Smaller models require less memory.</p>
        <p style="text-align:left;">4. <b>Abstractness:</b> Slide to select the level of abstractness of your document, vary this attribute to explore different images.</p>

        <p style="text-align:left;"> <b>For more details:</b> check out my <a href="https://fittar.me/post/summagary/" target="_blank">blog post</a> for a comprehensive explanation of the Summagery project.</p>
        """)


    inputs = [
        gr.Textbox(label="Document", lines=10,interactive=True),
        gr.Number(label="Batch Size", value=5),
        gr.Dropdown(label="T5_Model_Checkpoint", choices=["t5-large", "t5-base", "t5-small"], value='t5-large'),
        gr.Slider(label="Abstractness", minimum=0, maximum=1, value=.2)
    ]

    outputs = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
        , columns=[2], rows=[2], object_fit="contain", height="auto")

    clear = gr.ClearButton([inputs[0]])
    greet_btn = gr.Button("Submit")
    greet_btn.click(fn=generate, inputs=inputs, outputs=outputs, api_name="Summagery")

demo.launch(share=True)