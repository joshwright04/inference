#include <gtest/gtest.h>

#include "simple.h"

TEST(AddTest, AddsTwoNumbers) {
    EXPECT_EQ(add(2, 3), 5);
}
