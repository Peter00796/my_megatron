import struct

def float_to_bf16(decimal_number):
    # Pack the float into 32-bit binary representation
    packed_float = struct.pack('>f', decimal_number)
    # Convert the packed float to an integer
    int_rep = struct.unpack('>I', packed_float)[0]
    
    # Extract the sign bit, exponent, and mantissa from the 32-bit float
    sign = (int_rep >> 31) & 0x1
    exponent = (int_rep >> 23) & 0xFF
    mantissa = (int_rep >> 16) & 0x7F
    
    # Combine them into the bf16 format
    bf16_rep = (sign << 15) | (exponent << 7) | mantissa
    
    # Convert the bf16 representation to binary string
    bf16_binary = f'{bf16_rep:016b}'
    
    return bf16_binary

# Example usage:
decimal_number = 0.001953125
bf16_binary = float_to_bf16(decimal_number)
print(f'Decimal: {decimal_number} -> BF16 Binary: {bf16_binary}')