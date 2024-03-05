from django.shortcuts import render

from django.core.handlers.wsgi import WSGIRequest


def profile(request: WSGIRequest):
    return render(request, "users/profile.html", {"user": request.user})
