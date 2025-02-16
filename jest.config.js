module.exports = {
    "preset": "ts-jest",
    "testEnvironment": "node",
    "moduleNameMapper": {
    "^@/(.*)$": "<rootDir>/src/$1"
    },
    // Optionally, specify the pattern to find test files
    testMatch: ['**/tests/**/*.[jt]s?(x)', '**/?(*.)+(spec|test).[tj]s?(x)'],
  };
  