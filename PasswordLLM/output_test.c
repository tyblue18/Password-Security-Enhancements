#include <stdio.h>
void write_to_array(int index, int value) {
    int arr[10];
    arr[index] = value;
    return;
}

int main() {
    int n = read_from_array(0);
    int x = 1;
    int y = 2;
    for (int i = 1; i <= 10; i++)
    {
        write_to_array(i, x);
        write_to_array(i + 10, y);
    }
    return 0;