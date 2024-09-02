from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import AppUser


class CustomUserAdmin(UserAdmin):
    model = AppUser
    # 추가된 필드를 관리자 페이지에서 보이도록 설정
    fieldsets = UserAdmin.fieldsets + ((None, {"fields": ("summoner",)}),)
    add_fieldsets = UserAdmin.add_fieldsets + ((None, {"fields": ("summoner",)}),)


admin.site.register(AppUser, CustomUserAdmin)
