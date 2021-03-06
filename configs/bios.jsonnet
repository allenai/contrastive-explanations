local transformer_model = "roberta-large";
local transformer_dim = 1024;
local max_len = 512;

{
  dataset_reader:{
    type: "text_classification_json",
//    max_sequence_length: max_len,
    tokenizer: {
        type: "pretrained_transformer",
        model_name: transformer_model,
//        max_length: max_len
//        do_lowercase: true
    },
    token_indexers: {
        tokens: {
            type: "pretrained_transformer",
            model_name: transformer_model,
            max_length: max_len
        }
    }
  },
  train_data_path: "data/bios/train.jsonl",
  validation_data_path: "data/bios/dev.jsonl",
  test_data_path: "data/bios/test.jsonl",
  evaluate_on_test: true,
  "model": {
    "type": "encoder_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1,
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 16
    }
  },
  "trainer": {
    "num_epochs": 20,
    patience: 5,
    "cuda_device" : 0,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-6,
      "weight_decay": 0.1,
    },
    num_gradient_accumulation_steps: 2
  }
}
