from django.urls import path
from .views import signin, signout, signup

app_name = "auth"
urlpatterns = [
    path("signin/", signin, name="signin"),
    path("signup/", signup, name="signup"),
    path("signout/", signout, name="signout"),
]
