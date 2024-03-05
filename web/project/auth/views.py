from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from .forms import SigninForm, SignupForm


def signin(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = SigninForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("home")
            else:
                form.add_error(None, "Invalid username or password")

    return render(request, "auth/signin.html")


def signout(request):
    logout(request)

    return redirect("home")


def signup(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("auth:signin")

    return render(request, "auth/signup.html")
