from data_utils import denormalize

def predict(env, policies):
    obs = env.reset()
    preds = {}

    for v in env.vars:
        if v in obs:
            a, _ = policies[v].sample(obs[v])
            preds[v] = a.detach().cpu().numpy()

    return preds