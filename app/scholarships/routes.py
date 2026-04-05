from flask import render_template, jsonify
from . import scholarship_bp
from modules.text_generation import generate_gemini_scholarships

SCHOLARSHIPS = [
    {
        "id": 1,
        "title": "National Scholarship Portal (NSP)",
        "agency": "Government of India",
        "category": "Government",
        "eligibility": "Class 1 to PhD students from minority communities, SC/ST, and OBC. Varies by scheme.",
        "description": "A centralized portal for various scholarship schemes offered by Central Government, State Governments, and UTs.",
        "link": "https://scholarships.gov.in/",
        "amount": "Varies by scheme",
        "status": "Ongoing"
    },
    {
        "id": 2,
        "title": "HDFC Badhte Kadam Scholarship",
        "agency": "HDFC Bank",
        "category": "Private",
        "eligibility": "Students from Class 11 to Post-graduation. Minimum 60% in previous class.",
        "description": "Support for students from underprivileged backgrounds to continue their education.",
        "link": "https://www.buddy4study.com/page/hdfc-bank-parivartan-s-ecss-programme",
        "amount": "Up to ₹1,00,000",
        "status": "Available"
    },
    {
        "id": 3,
        "title": "Reliance Foundation Scholarships",
        "agency": "Reliance Foundation",
        "category": "Private",
        "eligibility": "Undergraduate and Postgraduate students in select fields. Merit and Means based.",
        "description": "Aims to support talented students to pursue world-class education in India.",
        "link": "https://www.scholarships.reliancefoundation.org/",
        "amount": "Up to ₹6,00,000",
        "status": "Upcoming"
    },
    {
        "id": 4,
        "title": "Pre-Matric Scholarships for Minorities",
        "agency": "Ministry of Minority Affairs",
        "category": "Government",
        "eligibility": "Students from Class 1 to 10 belonging to minority communities with 50% marks in previous exam.",
        "description": "Encouraging parents from minority communities to send their children to school and lighten the financial burden.",
        "link": "https://scholarships.gov.in/",
        "amount": "₹1,000 to ₹10,000 per annum",
        "status": "Open"
    },
    {
        "id": 5,
        "title": "Tata Scholarship for Cornell University",
        "agency": "Tata Trusts",
        "category": "Non-Government",
        "eligibility": "Indian students pursuing undergraduate studies at Cornell University.",
        "description": "Special fund for Indian students who demonstrate financial need.",
        "link": "https://admissions.cornell.edu/apply/international-students/tata-scholarship",
        "amount": "Full Tuition",
        "status": "Annual"
    },
    {
        "id": 6,
        "title": "INSPIRE Scholarship",
        "agency": "Department of Science and Technology",
        "category": "Government",
        "eligibility": "Top 1% in Class 12 board exams pursuing Basic/Natural Science courses (B.Sc/M.Sc).",
        "description": "Innovation in Science Pursuit for Inspired Research (INSPIRE) for talented youth.",
        "link": "https://online-inspire.gov.in/",
        "amount": "₹80,000 per annum",
        "status": "Open"
    },
    {
        "id": 7,
        "title": "PM YASASVI Scholarship",
        "agency": "Ministry of Social Justice & Empowerment",
        "category": "Government",
        "eligibility": "OBC, EBC, DNT students of Class 9 and 11 based on YET entrance exam.",
        "description": "Scholarship for Young Achievers Scholarship Award Scheme for Vibrant India.",
        "link": "https://yet.nta.ac.in/",
        "amount": "Up to ₹1,25,000 per annum",
        "status": "Open"
    },
    {
        "id": 8,
        "title": "Kotak Kanya Scholarship",
        "agency": "Kotak Education Foundation",
        "category": "Private",
        "eligibility": "Meritorious girl students pursuing 1st year professional graduation courses (Engineering, MBBS, Architecture, Design, LLB).",
        "description": "Empowering girl students from low-income families to pursue higher education.",
        "link": "https://www.buddy4study.com/page/kotak-kanya-scholarship",
        "amount": "₹1,50,000 per year",
        "status": "Available"
    },
    {
        "id": 9,
        "title": "LIC Golden Jubilee Scholarship",
        "agency": "Life Insurance Corporation of India",
        "category": "Non-Government",
        "eligibility": "Students who have passed Class 12 with 60% and are below a certain income threshold.",
        "description": "Aimed at providing better opportunities for higher education to meritorious students.",
        "link": "https://licindia.in/web/guest/scholarships",
        "amount": "₹20,000 per annum",
        "status": "Open"
    },
    {
        "id": 10,
        "title": "Sitaram Jindal Foundation Scholarship",
        "agency": "Jindal Foundation",
        "category": "Non-Government",
        "eligibility": "Class 11, 12, ITI, Diploma, Undergraduate, and Postgraduate students.",
        "description": "Merit-based scholarships for students pursuing various educational paths.",
        "link": "https://www.sitaramjindalfoundation.org/scholarships-for-students-in-india.php",
        "amount": "₹500 to ₹3,200 per month",
        "status": "Open"
    },
    {
        "id": 11,
        "title": "NMMS - National Means Merit Scholarship",
        "agency": "Department of School Education",
        "category": "Government",
        "eligibility": "Class 8 students with 55% marks and family income below ₹3.5 Lakh.",
        "description": "Scholarship for students in government schools to prevent dropout at Class 8.",
        "link": "https://scholarships.gov.in/",
        "amount": "₹12,000 per annum",
        "status": "Ongoing"
    },
    {
        "id": 12,
        "title": "Legrand Empowerment Scholarship",
        "agency": "Legrand India",
        "category": "Private",
        "eligibility": "Girl students following B.Tech, BE, B.Arch, or other undergraduate courses.",
        "description": "Promoting gender diversity in technical education.",
        "link": "https://www.buddy4study.com/page/legrand-empowerment-scholarship-program",
        "amount": "₹60,000 to ₹1,00,000 per year",
        "status": "Available"
    },
    {
        "id": 13,
        "title": "Adobe Women-in-Technology",
        "agency": "Adobe",
        "category": "Private",
        "eligibility": "Female students pursuing undergraduate or masters in Computer Science or related fields.",
        "description": "Encouraging women to pursue careers in computing and technology.",
        "link": "https://research.adobe.com/adobe-india-women-in-technology-scholarship/",
        "amount": "$2,500 (One-time)",
        "status": "Annual"
    },
    {
        "id": 14,
        "title": "L’Oréal India For Young Women In Science",
        "agency": "L’Oréal India",
        "category": "Private",
        "eligibility": "Girl students who have passed Class 12 with 85% in PCB/PCM and family income < ₹6L.",
        "description": "Supporting young women to pursue science as their career path.",
        "link": "https://www.loreal.com/en/india/articles/commitment/l-oreal-india-for-young-women-in-science-scholarships/",
        "amount": "₹2,50,000",
        "status": "Open"
    },
    {
        "id": 15,
        "title": "Post-Matric Scholarship for SC/ST",
        "agency": "State Government",
        "category": "Government",
        "eligibility": "Students of Class 11 to PhD belonging to SC or ST categories.",
        "description": "State-level financial assistance for higher education for marginalized communities.",
        "link": "https://scholarships.gov.in/",
        "amount": "Full fees + maintenance allowance",
        "status": "Ongoing"
    },
    {
        "id": 16,
        "title": "Central Sector Scheme of Scholarship (CSSS)",
        "agency": "Department of Higher Education",
        "category": "Government",
        "eligibility": "Class 12 passed students in top 20th percentile of their board pursuing regular degree.",
        "description": "Financial assistance to meritorious students from low-income families to meet day-to-day expenses while pursuing higher studies.",
        "link": "https://scholarships.gov.in/",
        "amount": "₹12,000 to ₹20,000 per annum",
        "status": "Open"
    },
    {
        "id": 17,
        "title": "AICTE Pragati Scholarship for Girls",
        "agency": "AICTE",
        "category": "Government",
        "eligibility": "Girl students admitted to 1st year of Degree/Diploma level course in AICTE approved institutions.",
        "description": "Providing opportunity to young women to further their education and prepare for a successful future.",
        "link": "https://www.aicte-india.org/schemes/students-development-schemes/pragati-scholarship-scheme",
        "amount": "₹50,000 per annum",
        "status": "Ongoing"
    },
    {
        "id": 18,
        "title": "AICTE Saksham Scholarship",
        "agency": "AICTE",
        "category": "Government",
        "eligibility": "Specially-abled students (disability > 40%) admitted to AICTE approved technical institutions.",
        "description": "Supporting specially-abled children to pursue technical education.",
        "link": "https://www.aicte-india.org/schemes/students-development-schemes/saksham-registration-policy",
        "amount": "₹50,000 per annum",
        "status": "Ongoing"
    },
    {
        "id": 19,
        "title": "Maulana Azad National Fellowship",
        "agency": "University Grants Commission",
        "category": "Government",
        "eligibility": "M.Phil/PhD students belonging to minority communities who have cleared UGC-NET or CSIR-NET.",
        "description": "Integrated five-year fellowship for minority students to pursue higher studies like M. Phil and Ph.D.",
        "link": "https://www.ugc.ac.in/manf/",
        "amount": "₹31,000 to ₹35,000 per month",
        "status": "Annual"
    },
    {
        "id": 20,
        "title": "Fulbright-Nehru Master’s Fellowships",
        "agency": "USIEF",
        "category": "Non-Government",
        "eligibility": "Highly motivated Indian students with at least 3 years of professional experience pursuing Master's in the US.",
        "description": "Designed for outstanding Indians to pursue a master’s degree program at select U.S. colleges and universities.",
        "link": "https://www.usief.org.in/Fulbright-Nehru-Masters-Fellowships.aspx",
        "amount": "Full Funding (Airfare, Tuition, Living)",
        "status": "Closed (Annual)"
    }
]

@scholarship_bp.route('/')
def index():
    return render_template('scholarships/index.html', scholarships=SCHOLARSHIPS)

@scholarship_bp.route('/api/generate', methods=['POST'])
def api_generate():
    try:
        new_scholarships = generate_gemini_scholarships()
        return jsonify({'success': True, 'scholarships': new_scholarships})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400
