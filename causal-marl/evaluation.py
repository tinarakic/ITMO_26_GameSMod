from data_utils import denormalize

def predict(env, policies):
    """
    Выполняет предсказание значений для всех переменных среды с помощью
    соответствующих policy (обученных нейронных сетей).

    Args:
        env: среда, содержащая список переменных (env.vars)
             и метод reset() для получения наблюдений.
        policies: dict
            Словарь вида {variable: policy}, где каждая
            policy поддерживает метод sample().

    Returns:
        dict: {variable: prediction}, где prediction представляет
              собой предсказанное значение переменной в формате NumPy массива.
    """
    obs = env.reset()
    preds = {}

    for v in env.vars:
        if v in obs:
            a, _ = policies[v].sample(obs[v])
            preds[v] = a.detach().cpu().numpy()

    return preds