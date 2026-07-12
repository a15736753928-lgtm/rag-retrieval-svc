"""MinIO 连通性测试脚本"""

import sys
import io
from minio import Minio
from minio.error import S3Error

ENDPOINT = "127.0.0.1:9000"
ACCESS_KEY = "admin"
SECRET_KEY = "12345678"
BUCKET = "rag-documents"
SECURE = False


def test_connection():
    """测试 MinIO 连接"""
    print("=" * 50)
    print("1. 测试 MinIO 连接...")
    try:
        client = Minio(
            endpoint=ENDPOINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=SECURE,
        )
        buckets = client.list_buckets()
        print(f"   [PASS] 连接成功! 现有 buckets: {[b.name for b in buckets]}")
        return client
    except S3Error as e:
        print(f"   [FAIL] 连接失败 (S3Error): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"   [FAIL] 连接失败: {e}")
        sys.exit(1)


def test_bucket(client: Minio):
    """测试 bucket 是否存在 / 创建"""
    print("=" * 50)
    print("2. 测试 Bucket...")
    try:
        exists = client.bucket_exists(BUCKET)
        if exists:
            print(f"   [PASS] Bucket '{BUCKET}' 已存在")
        else:
            client.make_bucket(BUCKET)
            print(f"   [PASS] Bucket '{BUCKET}' 创建成功")

        assert client.bucket_exists(BUCKET), "Bucket 创建后验证失败"
        return True
    except S3Error as e:
        print(f"   [FAIL] Bucket 操作失败: {e}")
        return False


def test_upload(client: Minio):
    """测试文件上传"""
    print("=" * 50)
    print("3. 测试文件上传...")
    test_data = b"Hello MinIO! This is a test file.\n" * 100
    object_key = "__test__/hello.txt"

    try:
        client.put_object(
            bucket_name=BUCKET,
            object_name=object_key,
            data=io.BytesIO(test_data),
            length=len(test_data),
            content_type="text/plain",
        )
        print(f"   [PASS] 上传成功: {object_key} ({len(test_data)} bytes)")
        return object_key, test_data
    except S3Error as e:
        print(f"   [FAIL] 上传失败: {e}")
        return None, None


def test_download(client: Minio, object_key: str, expected_data: bytes):
    """测试文件下载"""
    print("=" * 50)
    print("4. 测试文件下载...")

    try:
        response = client.get_object(BUCKET, object_key)
        data = response.read()
        response.close()
        response.release_conn()

        content_type = response.headers.get("Content-Type", "")

        if data == expected_data:
            print(f"   [PASS] 下载成功! 内容一致 ({len(data)} bytes), Content-Type: {content_type}")
            return True
        else:
            print(f"   [FAIL] 下载内容不一致! expected={len(expected_data)}, got={len(data)}")
            return False
    except S3Error as e:
        print(f"   [FAIL] 下载失败: {e}")
        return False


def test_list_objects(client: Minio):
    """测试列出对象"""
    print("=" * 50)
    print("5. 测试列出对象...")
    try:
        objects = list(client.list_objects(BUCKET, prefix="__test__/"))
        print(f"   [PASS] 找到 {len(objects)} 个测试对象:")
        for obj in objects:
            print(f"      - {obj.object_name} ({obj.size} bytes)")
        return True
    except S3Error as e:
        print(f"   [FAIL] 列出对象失败: {e}")
        return False


def test_delete(client: Minio, object_key: str):
    """测试删除文件"""
    print("=" * 50)
    print("6. 测试文件删除...")
    try:
        client.remove_object(BUCKET, object_key)
        print(f"   [PASS] 删除成功: {object_key}")

        # 验证已删除
        try:
            client.stat_object(BUCKET, object_key)
            print(f"   [WARN] 对象仍然存在?")
            return False
        except S3Error:
            print(f"   [PASS] 已验证对象已删除")
            return True
    except S3Error as e:
        print(f"   [FAIL] 删除失败: {e}")
        return False


def test_presigned_url(client: Minio, object_key: str):
    """测试预签名 URL"""
    print("=" * 50)
    print("7. 测试预签名 URL...")
    test_data = b"test presigned url"
    obj = "__test__/presigned_test.txt"

    try:
        client.put_object(BUCKET, obj, io.BytesIO(test_data), len(test_data))
        url = client.presigned_get_object(BUCKET, obj)
        print(f"   [PASS] 预签名 URL 生成成功")
        print(f"   URL (前80字符): {url[:80]}...")

        # 清理
        client.remove_object(BUCKET, obj)
        return True
    except S3Error as e:
        print(f"   [FAIL] 预签名 URL 失败: {e}")
        return False


def main():
    print()
    print("=== MinIO 连通性测试 ===")
    print(f"   Endpoint: {ENDPOINT}")
    print(f"   Bucket:   {BUCKET}")
    print(f"   Secure:   {SECURE}")
    print()

    results = {}

    # 1. 连接
    client = test_connection()
    results["连接"] = client is not None

    # 2. Bucket
    results["Bucket"] = test_bucket(client)

    # 3. 上传
    obj_key, data = test_upload(client)
    results["上传"] = obj_key is not None

    # 4. 下载
    if obj_key:
        results["下载"] = test_download(client, obj_key, data)

    # 5. 列出
    results["列出"] = test_list_objects(client)

    # 6. 预签名 URL
    results["预签名URL"] = test_presigned_url(client, obj_key or "__test__/presigned_test.txt")

    # 7. 删除（最后做）
    if obj_key:
        results["删除"] = test_delete(client, obj_key)

    # 汇总
    print()
    print("=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    all_ok = True
    for name, ok in results.items():
        status = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_ok = False
        print(f"   {status}  {name}")

    if all_ok:
        print()
        print("所有测试通过! MinIO 可用。")
    else:
        print()
        print("部分测试失败，请检查 MinIO 服务。")
        sys.exit(1)


if __name__ == "__main__":
    main()
