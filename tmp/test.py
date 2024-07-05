def requires_permission(permission):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not has_permission(permission):
                print("Permission denied")
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator

def has_permission(permission):
    # 模拟权限检查逻辑
    return permission == "admin"

@requires_permission("admin")
def delete_user(user_id):
    print(f"User {user_id} deleted")

delete_user(123)  # 有权限
delete_user(456)  # 无权限
