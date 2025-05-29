import os
import argparse
from cryptography.fernet import Fernet

def decrypt_file(input_file, output_file, key):
    """解密单个文件"""
    f = Fernet(key)
    
    with open(input_file, 'rb') as file:
        encrypted_data = file.read()
    
    decrypted_data = f.decrypt(encrypted_data)
    
    with open(output_file, 'wb') as file:
        file.write(decrypted_data)
    
    print(f"已解密: {input_file} -> {output_file}")

def process_directory(directory, key, ext):
    """解密目录中的所有文件"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                output_path = file_path[:-len(ext)]
                decrypt_file(file_path, output_path, key)

def read_key_from_file(key_file):
    """从文件中读取密钥"""
    with open(key_file, 'r') as file:
        return file.read().strip()

def main():
    parser = argparse.ArgumentParser(description='文件解密工具')
    parser.add_argument('-i', '--input', required=True, help='输入文件或目录')
    parser.add_argument('-o', '--output', help='输出文件或目录 (默认移除扩展名)')
    parser.add_argument('-k', '--key', help='解密密钥')
    parser.add_argument('--key-file', help='从文件读取解密密钥')
    parser.add_argument('-d', '--directory', action='store_true', help='处理整个目录')
    parser.add_argument('--ext', default='.encrypted', help='加密文件的扩展名 (默认 .encrypted)')
    
    args = parser.parse_args()
    
    # 读取密钥
    if args.key_file:
        key = read_key_from_file(args.key_file)
    elif args.key:
        key = args.key
    else:
        print("错误: 必须提供密钥或密钥文件")
        return
    
    if args.directory:
        process_directory(args.input, key, args.ext)
    else:
        if not args.input.endswith(args.ext):
            print(f"错误: 输入文件 {args.input} 没有使用 {args.ext} 扩展名")
            return
        
        output = args.output or args.input[:-len(args.ext)]
        decrypt_file(args.input, output, key)

if __name__ == "__main__":
    main()

