from django.contrib.auth import logout
from django.shortcuts import redirect


def signout(request):
    logout(request)

    return redirect("home")
