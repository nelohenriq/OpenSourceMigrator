import os
import re
import shutil
import subprocess
import ast
import difflib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI

class TaskType(Enum):
    REASONING = "reasoning"
    DETERMINISTIC = "deterministic"
    CREATIVE = "creative"
    GENERAL = "general"

@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: Optional[str] = None
    preferred_tasks: Optional[List[TaskType]] = None
    model: Optional[str] = None

class ProviderManager:
    def __init__(self):
        self.env_vars = self.load_environment_vars()
        self.providers = self.initialize_providers()
        self.default_provider = self.detect_default_provider()
        
    def load_environment_vars(self) -> Dict[str, str]:
        load_dotenv(override=True)
        return {
            'groq': os.getenv('GROQ_API_KEY'),
            'deepseek': os.getenv('DEEPSEEK_API_KEY'),
            'openai': os.getenv('OPENAI_API_KEY')
        }

    def initialize_providers(self) -> Dict[str, ProviderConfig]:
        return {
            'groq': ProviderConfig(
                name='groq',
                base_url='https://api.groq.com/openai/v1',
                api_key=self.env_vars['groq'],
                preferred_tasks=[TaskType.DETERMINISTIC, TaskType.GENERAL],
                model='mixtral-8x7b-32768'
            ),
            'deepseek': ProviderConfig(
                name='deepseek',
                base_url='https://api.deepseek.com/v1',
                api_key=self.env_vars['deepseek'],
                preferred_tasks=[TaskType.REASONING, TaskType.CREATIVE],
                model='deepseek-chat'
            ),
            'ollama': ProviderConfig(
                name='ollama',
                base_url=self.env_vars['ollama'] or 'http://localhost:11434/v1',
                api_key='ollama',
                preferred_tasks=[TaskType.GENERAL],
                model='llama2'
            ),
            'openai': ProviderConfig(
                name='openai',
                base_url='https://api.openai.com/v1',
                api_key=self.env_vars['openai'],
                preferred_tasks=[TaskType.GENERAL],
                model='gpt-4'
            )
        }

    def detect_default_provider(self) -> ProviderConfig:
        priorities = ['groq', 'deepseek', 'ollama', 'openai']
        for provider in priorities:
            if self.providers[provider].api_key or provider == 'ollama':
                return self.providers[provider]
        raise ValueError("No valid provider configuration found")

    def get_provider_for_task(self, task_type: TaskType) -> ProviderConfig:
        for provider in self.providers.values():
            if provider.preferred_tasks and task_type in provider.preferred_tasks:
                if provider.api_key or provider.name == 'ollama':
                    return provider
        return self.default_provider

class OpenSourceMigrator:
    def __init__(self, source_dir: str, output_dir: str, test_mode: bool = False):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.test_mode = test_mode
        self.provider_manager = ProviderManager()
        self.log = []
        self.test_results = []
        self.errors = []
        self.task_mappings = {}

    def detect_task_type(self, code_block: str) -> TaskType:
        """Heuristic detection of task types from code context"""
        code_block = code_block.lower()
        
        reasoning_keywords = ['analyze', 'reason', 'solve', 'conclude']
        deterministic_keywords = ['process', 'calculate', 'execute', 'transform']
        creative_keywords = ['generate', 'create', 'write', 'imagine']
        
        if any(kw in code_block for kw in reasoning_keywords):
            return TaskType.REASONING
        if any(kw in code_block for kw in deterministic_keywords):
            return TaskType.DETERMINISTIC
        if any(kw in code_block for kw in creative_keywords):
            return TaskType.CREATIVE
        return TaskType.GENERAL

    def process_client_initialization(self, node: ast.AST, content: str) -> str:
        """Replace client initialization with task-specific configuration"""
        original_code = ast.unparse(node)
        context_code = self.get_code_context(content, node)
        task_type = self.detect_task_type(context_code)
        
        provider = self.provider_manager.get_provider_for_task(task_type)
        new_code = (
            f"# Using {provider.name} for {task_type.value} tasks\n"
            f"client = OpenAI(\n"
            f"    base_url='{provider.base_url}',\n"
            f"    api_key='{provider.api_key}'\n"
            f")\n"
            f"model = '{provider.model}'"
        )
        
        self.task_mappings[task_type] = provider.name
        return content.replace(original_code, new_code)

    def get_code_context(self, content: str, node: ast.AST) -> str:
        """Get surrounding code context for task detection"""
        lines = content.split('\n')
        start_line = max(node.lineno - 3, 0)
        end_line = node.end_lineno + 2
        return '\n'.join(lines[start_line:end_line])

    def process_file(self, file_path: Path):
        """Process files with task-aware provider selection"""
        rel_path = file_path.relative_to(self.source_dir)
        dest_path = self.output_dir / rel_path
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
            modified = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'client':
                            if isinstance(node.value, ast.Call):
                                content = self.process_client_initialization(node, content)
                                modified = True
                                self.log.append(f"Updated client for {task_type} task in {rel_path}")
            
            if modified or self.test_mode:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, 'w', encoding='utf-8') as f:
                    f.write(content)

        except SyntaxError as e:
            self.errors.append(f"Syntax error in {rel_path}: {str(e)}")

    def analyze_directory(self):
        """Scan directory and identify files for processing"""
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.json', '.md')):
                    src_path = Path(root) / file
                    yield src_path

    def validate_syntax(self, file_path: Path) -> bool:
        """Validate Python syntax using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            return True
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {str(e)}")
            return False

    def run_tests(self) -> bool:
        """Run project tests and return success status"""
        test_commands = [
            ['pytest', 'tests/'],
            ['python', '-m', 'unittest', 'discover'],
            ['python', 'setup.py', 'test']
        ]

        success = True
        for cmd in test_commands:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.output_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
                self.test_results.append(f"Tests passed with {cmd}:\n{result.stdout}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                success = False
                error_msg = f"Test failed with {cmd}:\nError: {str(e)}"
                if hasattr(e, 'stderr') and e.stderr:
                    error_msg += f"\nDetails: {e.stderr}"
                self.test_results.append(error_msg)
        return success

    def generate_diff_report(self, original_path: Path, migrated_path: Path):
        """Generate diff between original and migrated files"""
        with open(original_path, 'r') as f1, open(migrated_path, 'r') as f2:
            original = f1.readlines()
            migrated = f2.readlines()

        diff = difflib.unified_diff(
            original,
            migrated,
            fromfile=str(original_path),
            tofile=str(migrated_path),
            lineterm=''
        )
        return '\n'.join(diff)

    def analyze_dependencies(self):
        """Check for dependency conflicts in requirements.txt"""
        req_file = Path(self.output_dir) / 'requirements.txt'
        if not req_file.exists():
            return

        with open(req_file, 'r') as f:
            requirements = f.read()

        if 'openai' not in requirements:
            with open(req_file, 'a') as f:
                f.write('\nopenai>=1.0.0\n')
                self.log.append("Added OpenAI dependency to requirements.txt")

    def dry_run(self):
        """Simulate migration without modifying files"""
        self.log.append("DRY RUN - No files will be modified")
        for file_path in self.analyze_directory():
            rel_path = file_path.relative_to(self.source_dir)
            with open(file_path, 'r') as f:
                content = f.read()

            original_content = content
            for pattern, replacement in self.replacements.items():
                content = re.sub(pattern, replacement, content)

            if content != original_content:
                diff = difflib.unified_diff(
                    original_content.splitlines(),
                    content.splitlines(),
                    fromfile=f"Original/{rel_path}",
                    tofile=f"Migrated/{rel_path}"
                )
                self.log.append(f"\nDiff for {rel_path}:\n" + '\n'.join(diff))

    def generate_report(self):
        """Enhanced report with task-provider mapping"""
        report = [
            "Migration Report",
            "================",
            f"Source directory: {self.source_dir}",
            f"Output directory: {self.output_dir}",
            "\nTask-Provider Mapping:",
            *[f"- {task.value}: {provider}" for task, provider in self.task_mappings.items()],
            "\nModifications:",
            *self.log,
            "\nTest Results:",
            *self.test_results,
            "\nErrors:",
            *self.errors
        ]
        return '\n'.join(report)

    def generate_report(self):
        """Enhanced report with task-provider mapping"""
        report = [
            "Migration Report",
            "================",
            f"Source directory: {self.source_dir}",
            f"Output directory: {self.output_dir}",
            "\nTask-Provider Mapping:",
            *[f"- {task.value}: {provider}" for task, provider in self.task_mappings.items()],
            "\nModifications:",
            *self.log,
            "\nTest Results:",
            *self.test_results,
            "\nErrors:",
            *self.errors
        ]
        return '\n'.join(report)

    def run_migration(self):
        """Execute migration with task-aware provider selection"""
        # Existing setup and processing logic
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

        for file_path in self.analyze_directory():
            self.process_file(file_path)

        # Validation and testing logic
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith('.py'):
                    self.validate_syntax(Path(root) / file)

        if self.test_mode:
            test_success = self.run_tests()
            # Update report with test status

        # Generate final report
        report = self.generate_report()
        print(report)
        # Save reports to file

if __name__ == "__main__":
    source = input("Enter source directory path: ")
    output = input("Enter output directory path: ")
    test_mode = input("Enable test mode? (y/n): ").lower() == 'y'

    migrator = OpenSourceMigrator(source, output, test_mode)
    
    try:
        migrator.run_migration()
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        if migrator.errors:
            print("Additional errors:", '\n'.join(migrator.errors))