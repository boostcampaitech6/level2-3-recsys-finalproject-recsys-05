from django.urls import path
from .views import signout

app_name = "auth"
urlpatterns = [
    path("signout/", signout, name="signout"),
]
