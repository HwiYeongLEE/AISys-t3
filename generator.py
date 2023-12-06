from transformers import Pipeline,PreTrainedTokenizerFast,BartForConditionalGeneration
import torch
import re

class Generator(Pipeline): 
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {} 
        if "max_length" in kwargs:
          preprocess_kwargs["max_length"] = kwargs["max_length"]
        if "num_beams" in kwargs:
          preprocess_kwargs["num_beams"] = kwargs["num_beams"]

        return preprocess_kwargs, {}, {}
    def preprocess(self, inputs, **kwargs):
           inputs = re.sub(r'[^A-Za-z가-힣,<>0-9:&# ]', '', inputs)
           inputs = "질문 생성: <unused0>"+inputs
           
           input_ids =  [self.tokenizer.bos_token_id] + self.tokenizer.encode(inputs) + [self.tokenizer.eos_token_id] 
           return {"inputs":torch.tensor([input_ids]),'max_length':kwargs['max_length'],'num_beams':kwargs['num_beams'] }

    def _forward(self, model_inputs):
            self.model.to(self.device)
            res_ids = self.model.generate(
                model_inputs['inputs'].to(self.device), 
                max_length=model_inputs['max_length'],
                num_beams=model_inputs['num_beams'],
                eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=[[self.tokenizer.unk_token_id]]
            )
            return {"logits": res_ids}

    def postprocess(self, model_outputs):
            a = self.tokenizer.batch_decode(model_outputs["logits"].tolist())[0]
            out_question = a.replace('<s>', '').replace('</s>', '')            
            return out_question

    def _inference(self,paragraph,**kwargs):
      input_ids = self.preprocess(paragraph,**kwargs)
      reds_ids = self._forward(input_ids)
      out_question = self.postprocess(reds_ids)
      return out_question

    def make_question(self, text, **kwargs):
      words = text.split(" ")
      frame_size = kwargs['frame_size']
      hop_length = kwargs['hop_length']
      steps = round((len(words)-frame_size)/hop_length) + 1
      outs = []
      for step in range(steps):
          try:
              script = " ".join(words[step*hop_length:step*hop_length+frame_size])
          except:
              script = " ".join(words[(1+step)*hop_length:]) 
          out = self._inference(script,**kwargs)
          if out.endswith("?"):
            outs.append(self._inference(script,**kwargs)) 
      return outs