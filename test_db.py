from db_operations import get_recent_voicemails, get_voicemail_by_id

# Test the database
print("🔍 Checking database records...")

# Get recent voicemails
recent = get_recent_voicemails(5)
print(f"📊 Found {len(recent)} voicemails in database")

for vm in recent:
    print(f"\n📧 Voicemail ID: {vm.id}")
    print(f"   From: {vm.sender_email}")
    print(f"   Subject: {vm.subject}")
    print(f"   Phone: {vm.phone_number}")
    print(f"   File: {vm.file_name}")
    print(f"   Created: {vm.created_at}")
    print(f"   Processed: {vm.processed_at}")
    if vm.transcribed_text:
        print(f"   Text: {vm.transcribed_text[:100]}...")
    else:
        print("   Text: Not transcribed yet")

print("\n✅ Database check complete!")