from django import forms
from django.contrib.auth.forms import UserCreationForm
from users.models import AppUser


class SignupForm(UserCreationForm):
    email = forms.EmailField(max_length=200, help_text="Required")

    class Meta:
        model = AppUser
        fields = ("username", "email", "password1", "password2")


class SigninForm(forms.Form):
    username = forms.CharField(max_length=200)
    password = forms.CharField(widget=forms.PasswordInput)
