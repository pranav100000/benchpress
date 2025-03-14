#!/usr/bin/env python
"""End-to-end test for benchpress evaluation commands with debug output."""

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple


class E2ETest:
    """End-to-end test for benchpress evaluation."""

    def __init__(self, verbose: bool = True):
        """Initialize the E2E test runner.
        
        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        
    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled.
        
        Args:
            message: The message to log
        """
        if self.verbose:
            print(message)
            
    def run_command(self, command: List[str]) -> Tuple[int, str, str]:
        """Run a command and return the exit code, stdout, and stderr.
        
        Args:
            command: The command to run as a list of strings
            
        Returns:
            A tuple of (exit_code, stdout, stderr)
        """
        self.log(f"Running command: {' '.join(command)}")
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        exit_code = process.returncode
        
        return exit_code, stdout, stderr
    
    def run_test(self, name: str, command: List[str], expected_strings: List[str], 
                 timeout: int = 60, max_examples: int = 1) -> bool:
        """Run a test and check if the expected strings are in the output.
        
        Args:
            name: The name of the test
            command: The command to run as a list of strings
            expected_strings: Strings that should be in the output
            timeout: Maximum time to run the command in seconds
            max_examples: Maximum number of examples to process
            
        Returns:
            True if the test passed, False otherwise
        """
        self.log(f"\n==== Running test: {name} ====")
        
        # Add --limit flag to restrict number of examples
        if "--limit" not in command:
            command.extend(["--limit", str(max_examples)])
        
        # Set a timeout for the command
        start_time = time.time()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read output incrementally to show progress
        stdout_lines = []
        stderr_lines = []
        
        while process.poll() is None:
            # Check if we've exceeded the timeout
            if time.time() - start_time > timeout:
                process.terminate()
                self.log(f"Test {name} timed out after {timeout} seconds")
                return False
            
            # Read from stdout and stderr
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    stdout_lines.append(line)
                    if self.verbose:
                        print(line, end="")
            
            if process.stderr:
                line = process.stderr.readline()
                if line:
                    stderr_lines.append(line)
                    if self.verbose:
                        print(f"STDERR: {line}", end="")
                        
            # Sleep briefly to avoid excessive CPU usage
            time.sleep(0.1)
        
        # Get any remaining output
        if process.stdout:
            for line in process.stdout:
                stdout_lines.append(line)
                if self.verbose:
                    print(line, end="")
                    
        if process.stderr:
            for line in process.stderr:
                stderr_lines.append(line)
                if self.verbose:
                    print(f"STDERR: {line}", end="")
        
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        exit_code = process.returncode
        
        # Check if all expected strings are in the output
        missing_strings = []
        for expected in expected_strings:
            if expected not in stdout and expected not in stderr:
                missing_strings.append(expected)
        
        # Determine if the test passed
        passed = exit_code == 0 and not missing_strings
        
        # Print the results
        if passed:
            self.log(f"✅ Test {name} passed")
            self.passed += 1
        else:
            self.log(f"❌ Test {name} failed")
            if exit_code != 0:
                self.log(f"   Exit code: {exit_code}")
            if missing_strings:
                self.log(f"   Missing expected strings: {missing_strings}")
            self.failed += 1
            
        return passed
        
    def run_all_tests(self) -> None:
        """Run all tests."""
        # Test 1: Basic Math500 evaluation
        self.run_test(
            name="Basic Math500 Evaluation",
            command=["python", "-m", "benchpress.cli", "evaluate", 
                    "--task", "math500", 
                    "--model", "openai:gpt-4",
                    "--debug",
                    "--limit", "1"],
            expected_strings=[
                "Example 1/1",
                "Question",
                "Model Response",
                "Reference Answer",
                "Extraction Details",
                "Method:",
                "Confidence:",
            ],
            timeout=120
        )
        
        # Test 2: AIME24 evaluation
        self.run_test(
            name="AIME24 Evaluation", 
            command=["python", "-m", "benchpress.cli", "evaluate", 
                    "--task", "aime24", 
                    "--model", "openai:gpt-4",
                    "--debug",
                    "--limit", "1"],
            expected_strings=[
                "Example 1/1",
                "Question",
                "Model Response",
                "Reference Answer",
                "Extraction Details",
                "Method:",
                "Confidence:",
            ],
            timeout=120
        )
        
        # Test 3: GPQA evaluation
        self.run_test(
            name="GPQA Evaluation",
            command=["python", "-m", "benchpress.cli", "evaluate", 
                    "--task", "gpqa", 
                    "--model", "openai:gpt-4",
                    "--debug",
                    "--limit", "1"],
            expected_strings=[
                "Example 1/1",
                "Question",
                "Model Response",
                "Reference Answer",
                "Extraction Details",
                "Method:",
                "Confidence:",
            ],
            timeout=120
        )
        
        # Test 4: Symbolic fraction handling test
        self.run_test(
            name="Symbolic Fraction Handling",
            command=["python", "test_math_extraction.py"],
            expected_strings=[
                "Best match: 108 (method=explicit_answer_marker, confidence=1.0)",
                "Pattern: explicit_answer_marker",
                "Confidence: 1.0",
            ],
            timeout=30
        )
        
        # Print summary
        total = self.passed + self.failed + self.skipped
        self.log(f"\n==== Test Summary ====")
        self.log(f"Passed: {self.passed}/{total}")
        self.log(f"Failed: {self.failed}/{total}")
        if self.skipped > 0:
            self.log(f"Skipped: {self.skipped}/{total}")
            
        # Exit with the appropriate code
        if self.failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end tests for benchpress")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of examples per test")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds per test")
    
    args = parser.parse_args()
    
    print(f"Running e2e tests with limit={args.limit} and timeout={args.timeout}s")
    
    # Create and run the tests
    runner = E2ETest(verbose=args.verbose)
    runner.run_all_tests()