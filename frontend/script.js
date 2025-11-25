// frontend/script.js

// 🚨 Important: This port must match the port your FastAPI server is running on (default is 8000)
const API_URL = "http://127.0.0.1:8000/analyze"; 

document.addEventListener('DOMContentLoaded', () => {
    const runButton = document.getElementById('run-analysis');
    const queryInput = document.getElementById('query');
    const filepathInput = document.getElementById('filepath');
    const reportContainer = document.getElementById('report-container');
    const sourcesContainer = document.getElementById('sources-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const personaButtons = document.querySelectorAll('.persona-btn');
    const inputStatus = document.getElementById('input-status');

    let currentPersona = 'DOCTOR'; // Default persona

    // --- 1. Input Validation and Button Enabling ---
    function validateInputs() {
        const query = queryInput.value.trim();
        const filepath = filepathInput.value.trim();
        const isValid = query.length > 5 && filepath.length > 5;
        
        runButton.disabled = !isValid;
        inputStatus.textContent = isValid ? '' : 'Query and File Path are required.';
    }

    // Attach validation to input events
    queryInput.addEventListener('input', validateInputs);
    filepathInput.addEventListener('input', validateInputs);

    // --- 2. Persona Toggle Logic ---
    personaButtons.forEach(button => {
        button.addEventListener('click', () => {
            currentPersona = button.dataset.persona;
            personaButtons.forEach(btn => {
                // Reset styles
                btn.classList.remove('active', 'bg-green-600', 'bg-blue-600', 'text-white');
                btn.classList.add('bg-gray-700', 'text-gray-400');
            });

            // Apply active styles
            button.classList.remove('bg-gray-700', 'text-gray-400');
            button.classList.add('active', 'text-white');
            if (currentPersona === 'DOCTOR') {
                button.classList.add('bg-green-600');
            } else {
                button.classList.add('bg-blue-600');
            }
            validateInputs(); // Re-validate
        });
    });

    // Initialize the default selection style
    document.getElementById('persona-doctor').click();


    // --- 3. API Call and Report Generation ---
    runButton.addEventListener('click', async () => {
        const userQuery = queryInput.value.trim();
        const filePath = filepathInput.value.trim();

        if (runButton.disabled) return;
        
        // Reset and Start Loading State
        reportContainer.innerHTML = '';
        sourcesContainer.innerHTML = '';
        loadingIndicator.classList.remove('hidden');
        runButton.disabled = true; // Disable during analysis

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_query: userQuery,
                    file_path: filePath,
                    persona: currentPersona,
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle HTTP errors
                reportContainer.innerHTML = renderError(`Error (${response.status}): ${data.detail || 'API request failed.'}`);
                return;
            }

            // Process and Render Report
            renderReport(data.report, currentPersona);
            renderSources(data.sources);

        } catch (error) {
            reportContainer.innerHTML = renderError(`Network Error: Ensure backend is running on ${API_URL}.`);
            console.error('Fetch error:', error);
        } finally {
            loadingIndicator.classList.add('hidden');
            runButton.disabled = false;
            validateInputs();
        }
    });


    // --- 4. Rendering Functions ---

    function renderError(message) {
        return `<div class="p-4 bg-red-900 border border-red-700 rounded-xl text-red-300 font-bold">🛑 ${message}</div>`;
    }


    function renderReport(markdownText, persona) {
        // Regex splits text by the bold headings (e.g., **Clinical Explanation and Summary**)
        const sections = markdownText.split(/(\*\*.*?\*\*)/).filter(s => s.trim() !== '');
        reportContainer.innerHTML = '';
        
        let currentSection = null;
        let cardColorClass = persona === 'DOCTOR' ? 'doctor-persona border-green-500' : 'patient-persona border-blue-500';

        sections.forEach(segment => {
            if (segment.startsWith('**') && segment.endsWith('**')) {
                // Start of a new card section
                if (currentSection) {
                    reportContainer.appendChild(currentSection);
                }
                
                currentSection = document.createElement('div');
                currentSection.className = `card p-6 rounded-xl shadow-md bg-gray-800 transition-colors duration-300 ${cardColorClass}`;
                
                const title = document.createElement('h3');
                title.className = 'text-xl font-bold mb-3 text-white';
                title.innerHTML = segment.replace(/\*/g, '').trim(); 
                currentSection.appendChild(title);
                
            } else if (currentSection) {
                // Content of the current card section
                const content = document.createElement('p');
                content.className = 'text-gray-300 whitespace-pre-line leading-relaxed'; 
                content.innerText = segment.trim();
                currentSection.appendChild(content);
            }
        });
        // Append the last section
        if (currentSection) {
            reportContainer.appendChild(currentSection);
        }
    }


    function renderSources(sources) {
        sourcesContainer.innerHTML = '';
        
        if (sources.length === 0) {
            sourcesContainer.innerHTML = renderError('Retrieval failed: No relevant context was found in the target file.');
            return;
        }

        const container = document.createElement('div');
        container.className = 'mt-12 p-6 bg-gray-800 rounded-xl shadow-inner border border-gray-700';
        container.innerHTML = `
            <h3 class="text-xl font-semibold text-white cursor-pointer mb-4" onclick="document.getElementById('sources-list-content').classList.toggle('hidden')">
                📄 Supporting Evidence (${sources.length} Nodes Used)
            </h3>
            <div id="sources-list-content" class="space-y-4 text-sm text-gray-300"></div>
        `;
        sourcesContainer.appendChild(container);
        const sourcesListContent = document.getElementById('sources-list-content');
        
        sources.forEach((source, index) => {
            const item = document.createElement('div');
            const color = source.type.includes('image') ? 'text-yellow-400' : 
                          source.type.includes('audio') ? 'text-purple-400' : 
                          'text-green-400';

            item.className = 'p-3 bg-gray-900 rounded-lg border border-gray-700';

            item.innerHTML = `
                <div class="flex justify-between items-center text-xs">
                    <span class="font-bold ${color} truncate mr-4">Source [${index + 1}]: ${source.file}</span>
                    <span class="text-gray-500">Type: ${source.type} | Score: ${source.score}</span>
                </div>
                <div class="mt-2 text-xs">
                    <button class="text-indigo-400 hover:text-indigo-300 transition-colors" 
                            onclick="alert('Full Snippet:\\n\\n${source.content.replace(/'/g, '').replace(/"/g, '')}')">
                        (View Full Retrieved Snippet)
                    </button>
                </div>
            `;
            sourcesListContent.appendChild(item);
        });
    }

});