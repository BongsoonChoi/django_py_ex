from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
# Create your views here.

def post_list(request):
    return render(request, 'blog/post_list.html',{})

def post_list_json(request):
    return JsonResponse({
        'message': '안녕 파이썬 장고',
        'items': ['파이썬', '장고', 'AWS', 'Azure'],
    }, json_dumps_params={'ensure_ascii': False})       # True로 바꾸면 한글이 깨지는 문제생김.


def post_list_json2(request):
    json_context = JsonResponse({
        'message': '안녕 파이썬 장고',
        'items': ['파이썬', '장고', 'AWS', 'Azure'],
    }, json_dumps_params={'ensure_ascii': True})
    return HttpResponse(
        json_context,
        content_type=u"application/json; charset=utf-8",
        status=200)
