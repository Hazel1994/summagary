from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from utils import mpnet_embed_class, get_concreteness, Collate_t5
from torch.utils.data import DataLoader
from utils import SentenceDataset


class Summagery:

    def __init__(self, t5_checkpoint, batch_size=5, abstractness=.4, max_d_length=1256, num_prompt=3, device='cuda'):

        # ViPE: Visualize Pretty-much Everything
        self.vipe_model = GPT2LMHeadModel.from_pretrained('fittar/ViPE-M-CTX7')
        vipe_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        vipe_tokenizer.pad_token = vipe_tokenizer.eos_token
        self.vipe_tokenizer = vipe_tokenizer

        # SDXL, load both base & refiner
        self.basexl = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.refinerxl = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.basexl.text_encoder_2,
            vae=self.basexl.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        self.device = device
        self.max_d_length = max_d_length  # maximum document length to handle before chunking
        self.final_document_length = 60
        self.num_prompt = num_prompt  # how many prompts to generate per document
        self.abstractness = abstractness  # to explore the prompts , just a handle from 0 to 1
        self.concreteness_dataset = './data/concreteness.csv'
        self.batch_size = batch_size
        # T5
        self.t5_model = AutoModelWithLMHead.from_pretrained(t5_checkpoint)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_checkpoint, model_max_length=max_d_length)
        self.collate_t5 = Collate_t5(self.t5_tokenizer)

        # for concrteness rating of the prompts
        data = pd.read_csv(self.concreteness_dataset, header=0,
                           delimiter='\t')
        self.word2score = {w: s for w, s in zip(data['WORD'], data['RATING'])}

    # for large documents, divide them into chunks with self.max_d_length size
    def document_preprocess(self, document):
        documents = []
        words = document.split()
        if len(words) <= self.max_d_length:
            return [document]

        start = 0
        while (len(words) > start):
            if len(words) > start + self.max_d_length:
                chunk = ' '.join(words[start:start + self.max_d_length])
            else:
                chunk = ' '.join(words[start:])

            start += self.max_d_length
            documents.append(chunk)

        return documents

    def t5_summarize(self, document):

        continue_summarization = True
        if len(document.split()) <= self.final_document_length:
            return document

        self.t5_model.to(self.device)

        documents = self.document_preprocess(document)

        if len(documents) > self.batch_size:

            # use batch inference to make things faster
            while (continue_summarization):
                dataset = SentenceDataset(documents)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_t5, num_workers=2)
                summaries = ''
                print('summarizing...')
                for text_batch, batch in tqdm(dataloader):
                    if batch.input_ids.shape[1] > 5:
                        max_length = int(batch.input_ids.shape[1] / 2)  # summarize the current chunk by half
                        if max_length < self.final_document_length:  # unless max_length is too short
                            max_length = self.final_document_length

                        batch = batch.to(self.device)
                        generated_ids = self.t5_model.generate(input_ids=batch.input_ids,
                                                               attention_mask=batch.attention_mask, num_beams=3,
                                                               max_length=max_length,
                                                               repetition_penalty=2.5,
                                                               length_penalty=1.0, early_stopping=True)
                        preds = \
                            [self.t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                             for g
                             in
                             generated_ids]
                        for pred in preds:
                            summaries = summaries + pred + '. '
                    else:
                        for chunk in text_batch:
                            summaries = summaries + chunk + '. '

                if len(summaries.split()) <= self.final_document_length:
                    continue_summarization = False
                    print('finished summarizing.')
                else:
                    documents = self.document_preprocess(summaries)
        else:

            # skip batch inference since we only have a few documents
            while (continue_summarization):
                summaries = ''
                print('summarizing...')
                for chunk in tqdm(documents):
                    if len(chunk.split()) > 2:
                        max_length = int(len(chunk.split()) / 2)  # summarize the current chunk by half
                        if max_length < self.final_document_length:  # unless max_length is too short
                            max_length = self.final_document_length

                        input_ids = self.t5_tokenizer.encode('summarize: ' + chunk, return_tensors="pt",
                                                             add_special_tokens=True, padding='longest',
                                                             max_length=self.max_d_length)
                        input_ids = input_ids.to(self.device)
                        generated_ids = self.t5_model.generate(input_ids=input_ids, num_beams=3, max_length=max_length,
                                                               repetition_penalty=2.5,
                                                               length_penalty=1.0, early_stopping=True)

                        pred = \
                        [self.t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g
                         in
                         generated_ids][0]
                        summaries = summaries + pred + '. '
                    else:
                        summaries = summaries + chunk + '. '

                if len(summaries.split()) <= self.final_document_length:
                    continue_summarization = False
                    print('finished summarizing.')
                else:
                    documents = self.document_preprocess(summaries)

        return summaries

    def vipe_generate(self, summary, do_sample=True, top_k=100, epsilon_cutoff=.00005, temperature=1):
        batch_size = random.choice([20, 40, 60])
        input_text = [summary] * batch_size
        # mark the text with special tokens
        input_text = [self.vipe_tokenizer.eos_token + i + self.vipe_tokenizer.eos_token for i in input_text]
        batch = self.vipe_tokenizer(input_text, padding=True, return_tensors="pt")

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        self.vipe_model.to(self.device)
        # how many new tokens to generate at max
        max_prompt_length = 50

        generated_ids = self.vipe_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                 max_new_tokens=max_prompt_length, do_sample=do_sample, top_k=top_k,
                                                 epsilon_cutoff=epsilon_cutoff, temperature=temperature)
        # return only the generated prompts
        prompts = self.vipe_tokenizer.batch_decode(generated_ids[:, -(generated_ids.shape[1] - input_ids.shape[1]):],
                                                   skip_special_tokens=True)

        # for semantic similarity
        mpnet_object = mpnet_embed_class(device=self.device, nli=False)

        similarities = mpnet_object.get_mpnet_embed_batch(prompts, [summary] * batch_size,
                                                          batch_size=batch_size).cpu().numpy()
        concreteness_score = get_concreteness(prompts, self.word2score)

        final_scores = [i * (1 - self.abstractness) + (self.abstractness) * j for i, j in
                        zip(similarities, concreteness_score)]
        # Get the indices that would sort the final_scores in descending order
        sorted_indices = np.argsort(final_scores)[::-1]

        # Extract the indices of the top 5 highest scores
        top_indices = sorted_indices[:self.num_prompt]
        prompts = [prompts[i] for i in top_indices]

        return prompts

    def sdxl_generate(self, prompts):
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 50
        high_noise_frac = 0.8
        self.basexl.to(self.device)
        self.refinerxl.to(self.device)

        images=[]
        for i, p in enumerate(prompts):
            # torch.manual_seed(i)
            image = self.basexl(
                prompt=p,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type="latent",
            ).images
            image = self.refinerxl(
                prompt=p,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
            ).images[0]

            images.append(image)

        return images

    def ignite(self, document):
        prompts = []
        summary = self.t5_summarize(document)
        prompts.append(summary)
        summary = summary.replace('. ', '; ')
        print(summary)
        prompts.extend(self.vipe_generate(summary))

        for p in prompts:
            print(p + '\n')

        images=self.sdxl_generate(prompts)

        return images