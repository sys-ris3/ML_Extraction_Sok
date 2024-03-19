from encrypt_model import AESCipher


def testAES():
    key = 'DJM2GUH7TPNCRU4NVEJXX7QJE3BDBH'
    cipher = AESCipher(key)
    text = 'Some Text'
    encrypted = cipher.encrypt(text)

    decrypted = cipher.decrypt(encrypted)
    if decrypted != text:
        raise Exception('Match error: "%s" != "%s"' % (text, decrypted))


def main():
    print('\n-----------------------------------')
    print('Tests:')

    tests = [testAES]

    for test in tests:
        test()
        print('Test "%s" is succeed\n' % test.func_name)

    print('-----------------------------------\n')


if __name__ == "__main__":
    main()
