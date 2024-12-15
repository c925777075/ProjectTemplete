from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2Model
import  torch

model_name = "/mnt/i/myai/MyLab/pretrained_models/Qwen2.5-0.5B-Instruct"

model = Qwen2Model.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Give me a short introduction to large language model."
text2 = "hello, hello。"

model_inputs = tokenizer([text, text2],
                         truncation=True,
                         padding=True,
                         return_tensors='pt').to(model.device)
text_hidden_state = model(**model_inputs)[0]
padding_mask = model_inputs['attention_mask']
bs = text_hidden_state.shape[0]
for b in range(bs):
    text_hidden_state_unpad = text_hidden_state[b][model_inputs['attention_mask'][b].type(torch.bool)]
    text_hidden_state_unpad.mean(dim=0)