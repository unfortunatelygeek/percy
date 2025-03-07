# Comprehensive Report on Data Collection for Intelligent Test Selection

## 1. Overview

Data collection was the most critical phase in building our Intelligent Test Selection system. Since the goal was to train an ML model to predict which test cases should run based on code changes, we needed a high-quality dataset from CI/CD logs and Git diffs. However, collecting this data was far from straightforward.

This report details:

- How we identified relevant GitHub repositories
- Challenges faced in obtaining public CI/CD logs
- Methods used to extract structured data from GitHub workflows
- Final dataset structure and its intended use

## 2. Challenges Faced During Data Collection

### A. Identifying Public React Projects with CI/CD Logs

Since we wanted to train our model on real-world data, we needed open-source React projects that actively used GitHub Actions for CI/CD. However, the process was challenging:

1. Many repositories either lacked CI/CD integration or used third-party services (Jenkins, CircleCI, TravisCI) instead of GitHub Actions.

2. Projects often deleted their workflow logs, meaning we couldn’t get historical test results.

3. Private repositories prevented access to workflow logs, limiting our dataset.

4. Incomplete test execution data – Some logs only contained test pass/fail summaries without execution times or error messages.

### B. Extracting Meaningful Data from GitHub Actions Logs

Even after identifying suitable projects, parsing raw logs was difficult due to:

1. Unstructured log formats – Output varied significantly between repositories.

2. Test frameworks varied (Jest, Mocha, Vitest, etc.), making it hard to standardize pass/fail analysis.

3. Interleaved log messages – Non-test-related logs (e.g., package installation, linting) cluttered the output.

### C. Mapping Test Failures to Code Changes

There was no direct mapping between a commit and the specific tests affected by that change.

Many repositories ran all tests indiscriminately, making it difficult to determine if a failure was related to a specific code change or an unrelated issue.

Flaky tests distorted failure patterns, making it hard to distinguish real failures from environment-related ones.

## 3. Methodology for Data Collection

### A. Searching for Open-Source Projects with CI/CD

To ensure high-quality data, we applied the following filtering criteria:

1. Used GitHub Actions (excluding projects using TravisCI, Jenkins, or CircleCI)
2. Had a React-based frontend (to align with Pax's target use cases)
3. Maintained at least 6 months of commit history
4. Had test logs that weren’t deleted
5. Ran at least 50 test cases per execution (ensuring meaningful test execution patterns)
6. Actively maintained (recent commits within the last 3 months)

The scraping script for the same will be made available shortly.

### B. Extracting Test Execution Data from GitHub Workflows

To process GitHub Actions logs, we:

1. Used GitHub’s API to retrieve workflow runs and logs.
2. Filtered logs to extract only test-related outputs.
3. Parsed test execution details including:
 - Test names and status (pass/fail)
 - Execution time per test
 - Error messages for failed tests
 - Flaky test detection based on multiple runs
4. Stored extracted data in a structured format (JSON/CSV) for ML training.

### C. Linking Git Diffs to Test Failures

To establish a relationship between code changes and test results, we:

1. Collected Git diffs for commits that triggered a CI run.

2. Used Tree-sitter to extract structured features from diffs:
 - Modified functions, classes, and files
 - Complexity of changes (e.g., simple refactor vs. logic changes)
 - Dependency impact (how changes propagate in the codebase)
 - Mapped test failures to recent code changes, identifying high-risk modifications.

## 4. Extracted Data and Its Intended Use

After completing data collection and preprocessing, we extracted the following structured features:

### A. Code Change Metadata (Git Diffs + AST Analysis)

- Modified files

- Functions & classes changed

- Number of lines added/removed

- Complexity score of changes

- Dependency graph impact

### B. Test Case Information (From GitHub Actions Logs)

- Test names and execution status (pass/fail)

- Execution time per test

- Flakiness score (based on historical test failures)

- Test failure impact (whether failure blocks deployment or is minor)

### C. Risk-Based Prioritization Factors

- Historical failure probability for modified files/functions

- Security vulnerability markers (if change affects a high-risk area)

- Code churn rate (how often a function is modified)

### D. Intended Use in Intelligent Test Selection

This dataset will be used to train an ML model that:

- Avoids running unnecessary tests → Reducing CI execution time.

- Prioritizes high-risk tests first → Catching critical failures early.

- Adapts dynamically → Learning from past test execution patterns.

- Handles flaky tests intelligently → Running unreliable tests only when necessary.

- Integrates with Pax → Providing real-time feedback to developers on which tests are most relevant to their changes.