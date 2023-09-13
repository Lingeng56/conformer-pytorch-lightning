import sys

sys.path.append('/home/wuliu/workspace/conformer-pytorch-lightning')

from django.http import JsonResponse
from src.deploy import build_model, preprocess
from django.views.decorators.csrf import ensure_csrf_cookie

model = build_model()
model.eval()


# Create your views here.
@ensure_csrf_cookie
def recognize(request):
    try:
        audio = request.FILES['audio'].read()
        feat, feat_length = preprocess(audio)
        hyps = model.model.greedy_search(feat, feat_length)

        content = []
        for w in hyps:
            if w == model.model.eos:
                break
            content.append(model.char_dict[w])
        text = model.sp.decode(content)
        result = {"message": text,
                  "status": "success"}
    except Exception as e:
        result = {
            'status': "fail",
            "message": str(e)
        }
    return JsonResponse(result)
