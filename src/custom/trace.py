from src.utils.causal_trace import *

def calculate_hidden_flow_v2(
    mt, prompt, subject, target, samples=10, noise=0.1, window=10, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    Takes an extra parameter target and asks for the reconstruction of target. 
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        # answer_t, best_score = [d[0] for d in predict_from_input(mt.model, inp)]
        answer_t   = mt.tokenizer(target, return_tensors="pt")["input_ids"][0][0].to(device='cuda') # [d[0] for d in predict_from_input(mt.model, inp)]
        scores = mt.model(answer_t.reshape(1,1)).logits[:, -1, :]
        base_score = scores[0][answer_t]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )
    

def plot_hidden_flow_v2(
    mt,
    prompt,
    subject=None,
    target=None,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow_v2(
        mt, prompt, subject, target, samples=samples, noise=noise, window=window, kind=kind
    )
    plot_trace_heatmap(result, savepdf, modelname=modelname)


def plot_all_flow_v2(mt, prompt, subject=None, target=None, noise=0.1, modelname=None):
    for kind in [None, "mlp", "attn"]:
        plot_hidden_flow_v2(
            mt, prompt, subject, target, modelname=modelname, noise=noise, kind=kind
        )
