from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect, reverse
from .forms import SigninForm, SignupForm


def signin(request):
    _next = request.GET.get("next")

    if request.user.is_authenticated:
        if _next:
            return redirect(_next)
        else:
            return redirect("home")

    if request.method == "POST":
        form = SigninForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                if _next:
                    return redirect(_next)
                else:
                    return redirect("home")
            else:
                form.add_error(None, "Invalid username or password")

    return render(request, "auth/signin.html", {"next": _next})


def signout(request):
    logout(request)

    return redirect("home")


def signup(request):
    _next = request.GET.get("next")

    if request.user.is_authenticated:
        if _next:
            return redirect(_next)
        else:
            return redirect("home")

    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(reverse("auth:signin") + f"?next={_next}")

    return render(request, "auth/signup.html", {"next": _next})
