"""
🔑 API Key Validation Script for SteadyPrice Week 8

This script validates that all required API keys are properly configured
and working for the multi-agent system.
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Try to import API libraries
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic library not installed. Run: pip install anthropic")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not installed. Run: pip install openai")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIKeyValidator:
    """Validates API keys for all required services."""
    
    def __init__(self):
        self.validation_results = {}
        
    def load_api_keys(self) -> Dict[str, Optional[str]]:
        """Load API keys from environment variables."""
        return {
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "amazon_access": os.getenv("AMAZON_ACCESS_KEY"),
            "amazon_secret": os.getenv("AMAZON_SECRET_KEY"),
            "bestbuy_key": os.getenv("BEST_BUY_API_KEY"),
            "bestbuy_secret": os.getenv("BEST_BUY_API_SECRET"),
            "walmart_key": os.getenv("WALMART_API_KEY")
        }
    
    async def validate_anthropic_api(self, api_key: str) -> Dict[str, Any]:
        """Validate Anthropic Claude API key."""
        if not ANTHROPIC_AVAILABLE:
            return {
                "status": "skipped",
                "reason": "Anthropic library not installed",
                "install_command": "pip install anthropic"
            }
        
        if not api_key:
            return {
                "status": "error",
                "reason": "ANTHROPIC_API_KEY not found in environment variables"
            }
        
        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            
            # Test with a simple message
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": "Respond with just 'OK'"
                }]
            )
            
            return {
                "status": "success",
                "model": "claude-3-5-sonnet-20241022",
                "response": response.content[0].text.strip(),
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
        except anthropic.AuthenticationError as e:
            return {
                "status": "error",
                "reason": f"Authentication failed: {str(e)}"
            }
        except anthropic.RateLimitError as e:
            return {
                "status": "error",
                "reason": f"Rate limit exceeded: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": f"Unexpected error: {str(e)}"
            }
    
    async def validate_openai_api(self, api_key: str) -> Dict[str, Any]:
        """Validate OpenAI API key."""
        if not OPENAI_AVAILABLE:
            return {
                "status": "skipped",
                "reason": "OpenAI library not installed",
                "install_command": "pip install openai"
            }
        
        if not api_key:
            return {
                "status": "error",
                "reason": "OPENAI_API_KEY not found in environment variables"
            }
        
        try:
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # Test with a simple completion
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Respond with just 'OK'"}],
                max_tokens=10
            )
            
            return {
                "status": "success",
                "model": "gpt-4o-mini",
                "response": response.choices[0].message.content.strip(),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except openai.AuthenticationError as e:
            return {
                "status": "error",
                "reason": f"Authentication failed: {str(e)}"
            }
        except openai.RateLimitError as e:
            return {
                "status": "error",
                "reason": f"Rate limit exceeded: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": f"Unexpected error: {str(e)}"
            }
    
    def validate_retailer_apis(self, api_keys: Dict[str, str]) -> Dict[str, Any]:
        """Validate retailer API keys (basic validation)."""
        retailer_results = {}
        
        # Amazon API
        if api_keys["amazon_access"] and api_keys["amazon_secret"]:
            retailer_results["amazon"] = {
                "status": "configured",
                "access_key": api_keys["amazon_access"][:10] + "..." if api_keys["amazon_access"] else None,
                "secret_key": "***configured***" if api_keys["amazon_secret"] else None
            }
        else:
            retailer_results["amazon"] = {
                "status": "missing",
                "reason": "Amazon API keys not found"
            }
        
        # Best Buy API
        if api_keys["bestbuy_key"] and api_keys["bestbuy_secret"]:
            retailer_results["bestbuy"] = {
                "status": "configured",
                "api_key": api_keys["bestbuy_key"][:10] + "..." if api_keys["bestbuy_key"] else None
            }
        else:
            retailer_results["bestbuy"] = {
                "status": "missing",
                "reason": "Best Buy API keys not found"
            }
        
        # Walmart API
        if api_keys["walmart_key"]:
            retailer_results["walmart"] = {
                "status": "configured",
                "api_key": api_keys["walmart_key"][:10] + "..." if api_keys["walmart_key"] else None
            }
        else:
            retailer_results["walmart"] = {
                "status": "missing",
                "reason": "Walmart API key not found"
            }
        
        return retailer_results
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive API key validation."""
        logger.info("🔑 Starting API Key Validation...")
        
        api_keys = self.load_api_keys()
        validation_start = datetime.utcnow()
        
        results = {
            "validation_summary": {
                "start_time": validation_start.isoformat(),
                "total_services": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0
            },
            "detailed_results": {}
        }
        
        # Validate Anthropic API
        logger.info("🤖 Validating Anthropic Claude API...")
        anthropic_result = await self.validate_anthropic_api(api_keys["anthropic"])
        results["detailed_results"]["anthropic"] = anthropic_result
        results["validation_summary"]["total_services"] += 1
        
        if anthropic_result["status"] == "success":
            results["validation_summary"]["successful"] += 1
        elif anthropic_result["status"] == "skipped":
            results["validation_summary"]["skipped"] += 1
        else:
            results["validation_summary"]["failed"] += 1
        
        # Validate OpenAI API
        logger.info("🧠 Validating OpenAI API...")
        openai_result = await self.validate_openai_api(api_keys["openai"])
        results["detailed_results"]["openai"] = openai_result
        results["validation_summary"]["total_services"] += 1
        
        if openai_result["status"] == "success":
            results["validation_summary"]["successful"] += 1
        elif openai_result["status"] == "skipped":
            results["validation_summary"]["skipped"] += 1
        else:
            results["validation_summary"]["failed"] += 1
        
        # Validate retailer APIs
        logger.info("🏪 Validating Retailer APIs...")
        retailer_results = self.validate_retailer_apis(api_keys)
        results["detailed_results"]["retailers"] = retailer_results
        
        # Count retailer API results
        for retailer, result in retailer_results.items():
            results["validation_summary"]["total_services"] += 1
            if result["status"] == "configured":
                results["validation_summary"]["successful"] += 1
            else:
                results["validation_summary"]["failed"] += 1
        
        # Calculate success rate
        total = results["validation_summary"]["total_services"]
        successful = results["validation_summary"]["successful"]
        success_rate = (successful / total * 100) if total > 0 else 0
        
        results["validation_summary"]["success_rate"] = success_rate
        results["validation_summary"]["end_time"] = datetime.utcnow().isoformat()
        results["validation_summary"]["duration_seconds"] = (
            datetime.utcnow() - validation_start
        ).total_seconds()
        
        return results
    
    def generate_setup_instructions(self, results: Dict[str, Any]) -> str:
        """Generate setup instructions based on validation results."""
        instructions = []
        
        # Check Anthropic
        anthropic_result = results["detailed_results"].get("anthropic", {})
        if anthropic_result.get("status") == "error":
            instructions.append("""
🤖 ANTHROPIC CLAUDE API SETUP:
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create new API key
5. Set environment variable:
   export ANTHROPIC_API_KEY=sk-ant-your-key-here
6. Install library: pip install anthropic
""")
        
        # Check OpenAI
        openai_result = results["detailed_results"].get("openai", {})
        if openai_result.get("status") == "error":
            instructions.append("""
🧠 OPENAI API SETUP:
1. Go to https://platform.openai.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create new secret key
5. Set environment variable:
   export OPENAI_API_KEY=sk-your-key-here
6. Install library: pip install openai
""")
        
        # Check retailer APIs
        retailer_results = results["detailed_results"].get("retailers", {})
        missing_retailers = [name for name, result in retailer_results.items() 
                           if result.get("status") == "missing"]
        
        if missing_retailers:
            instructions.append(f"""
🏪 RETAILER API SETUP (Optional):
Missing APIs: {', '.join(missing_retailers)}

For Amazon Product Advertising API:
1. Go to https://affiliate-program.amazon.com/
2. Sign up for the program
3. Create Product Advertising API credentials
4. Set environment variables:
   export AMAZON_ACCESS_KEY=your-access-key
   export AMAZON_SECRET_KEY=your-secret-key
   export AMAZON_ASSOCIATE_TAG=your-tag

For Best Buy Developer API:
1. Go to https://developer.bestbuy.com/
2. Sign up for developer account
3. Create API keys
4. Set environment variables:
   export BEST_BUY_API_KEY=your-key
   export BEST_BUY_API_SECRET=your-secret

For Walmart Developer API:
1. Go to https://developer.walmart.com/
2. Sign up for developer account
3. Create API key
4. Set environment variable:
   export WALMART_API_KEY=your-key
""")
        
        return "\n".join(instructions)

async def main():
    """Main validation function."""
    print("🔑 SteadyPrice Week 8 - API Key Validation")
    print("=" * 60)
    
    validator = APIKeyValidator()
    results = await validator.run_full_validation()
    
    # Display results
    summary = results["validation_summary"]
    print(f"\n📊 Validation Summary:")
    print(f"   Total Services: {summary['total_services']}")
    print(f"   ✅ Successful: {summary['successful']}")
    print(f"   ❌ Failed: {summary['failed']}")
    print(f"   ⏭️  Skipped: {summary['skipped']}")
    print(f"   📈 Success Rate: {summary['success_rate']:.1f}%")
    print(f"   ⏱️  Duration: {summary['duration_seconds']:.2f}s")
    
    print(f"\n🔍 Detailed Results:")
    print("-" * 40)
    
    # Anthropic results
    anthropic_result = results["detailed_results"].get("anthropic", {})
    status_icon = "✅" if anthropic_result.get("status") == "success" else "❌" if anthropic_result.get("status") == "error" else "⏭️"
    print(f"{status_icon} Anthropic Claude API: {anthropic_result.get('status', 'unknown')}")
    if anthropic_result.get("status") == "success":
        print(f"   Model: {anthropic_result.get('model')}")
        print(f"   Response: {anthropic_result.get('response')}")
    elif anthropic_result.get("status") == "error":
        print(f"   Error: {anthropic_result.get('reason')}")
    elif anthropic_result.get("status") == "skipped":
        print(f"   Reason: {anthropic_result.get('reason')}")
        print(f"   Install: {anthropic_result.get('install_command')}")
    
    # OpenAI results
    openai_result = results["detailed_results"].get("openai", {})
    status_icon = "✅" if openai_result.get("status") == "success" else "❌" if openai_result.get("status") == "error" else "⏭️"
    print(f"{status_icon} OpenAI API: {openai_result.get('status', 'unknown')}")
    if openai_result.get("status") == "success":
        print(f"   Model: {openai_result.get('model')}")
        print(f"   Response: {openai_result.get('response')}")
    elif openai_result.get("status") == "error":
        print(f"   Error: {openai_result.get('reason')}")
    elif openai_result.get("status") == "skipped":
        print(f"   Reason: {openai_result.get('reason')}")
        print(f"   Install: {openai_result.get('install_command')}")
    
    # Retailer results
    retailer_results = results["detailed_results"].get("retailers", {})
    print(f"\n🏪 Retailer APIs:")
    for retailer, result in retailer_results.items():
        status_icon = "✅" if result.get("status") == "configured" else "❌"
        print(f"{status_icon} {retailer.title()}: {result.get('status')}")
        if result.get("status") == "missing":
            print(f"   Reason: {result.get('reason')}")
    
    # Generate setup instructions if needed
    if summary["failed"] > 0 or summary["skipped"] > 0:
        print(f"\n🔧 Setup Instructions:")
        print("=" * 60)
        instructions = validator.generate_setup_instructions(results)
        print(instructions)
    
    # Save results
    with open("API_KEY_VALIDATION_REPORT.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: API_KEY_VALIDATION_REPORT.json")
    
    # Final recommendation
    if summary["successful"] >= 2:  # At least Anthropic and OpenAI working
        print(f"\n🎉 SUCCESS! Your API keys are configured for full Week 8 functionality!")
        print(f"   You can now run the complete empirical validation with all features enabled.")
        print(f"   Expected improvements:")
        print(f"   • FrontierAgent: Will pass with actual API performance")
        print(f"   • Overall system: Higher success rate and better metrics")
        print(f"   • Business impact: Real API costs and performance data")
    else:
        print(f"\n⚠️  ACTION REQUIRED! Configure API keys for full functionality.")
        print(f"   Follow the setup instructions above to enable all Week 8 features.")

if __name__ == "__main__":
    asyncio.run(main())
