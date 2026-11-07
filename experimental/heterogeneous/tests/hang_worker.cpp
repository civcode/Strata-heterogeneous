// Test-only: ignores control socket shutdown and forces supervisor timeout.
#include <unistd.h>
int main() {
    for (;;) ::sleep(10);
}
