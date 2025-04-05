import argparse

def protgpt_wrapper(samples, model, tokenizer):
    ppls = []
    for seq in samples:
        out = tokenizer(seq, return_tensors="pt")
        input_ids = out.input_ids.cuda()

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        ppl = (outputs.loss * input_ids.shape[1]).item()
        ppls.append(ppl)
    
    ppls = np.array(ppls)
    return ppls

def extract_ll_distr(df, seq_label, model, tokenizer):
    sequences = df[seq_label]
    return -1 * protgpt_wrapper(sequences, model, tokenizer)

def extract_ll_directory(dir_name, seq_label, model, tokenizer):
    for fn in os.listdir(dir_name):
        file_path = os.path.join(dir_name, fn)
        if fn.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_path, nrows=0)  # Read only the header
                if 'loglikelihood' in df.columns: # Don't re-compute if already has loglikelihood
                    print(f"{fn} already processed - Skipping...")
                elif seq_label in df.columns:
                    df = pd.read_csv(file_path)
                    ll_distr = extract_ll_distr(df, seq_label, model, tokenizer)
                    df['loglikelihood'] = ll_distr
                    df.to_csv(file_path, index=False)
            except Exception as e: 
                pass
