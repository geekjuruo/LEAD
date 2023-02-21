from torch import nn
import transformers.models.bert.modeling_bert

from utils._global import _global


class SpellBERT(nn.Module):
    def __init__(self):
        # Components
        super(SpellBERT, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(_global.config.get("model", "pretrained_model"))
        config: transformers.BertConfig = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

        # Constants
        self.keys = ["input_ids", "attention_mask", "token_type_ids", "inputs_embeds"]

    def forward(self, **kwargs):
        inputs = {key: value for key, value in kwargs.items() if key in self.keys}
        outputs = self.bert(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(self.dropout(sequence_output))
        if kwargs.get("return_output", None):
            return logits, (sequence_output, outputs.hidden_states)
        if kwargs.get("output_attentions", None):
            return logits, outputs.attentions
        return logits