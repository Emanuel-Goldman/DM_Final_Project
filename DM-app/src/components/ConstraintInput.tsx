import { Alert, Button, SpaceBetween, Textarea, TextareaProps} from "@cloudscape-design/components";
import { useState } from "react";

// Define a type for constraint items as a tuple: [weight1, weight2, operator]
type ConstraintItem = [number, number, string];

export const ConstraintInput: React.FC = () => {
    const [constraints, setConstraints] = useState<string>("");
    const [parsedConstraints, setParsedConstraints] = useState<ConstraintItem[]>([]);
    const [error, setError] = useState<string | null>(null);

    const handleConstraintsChange: TextareaProps['onChange'] = (event) => {
        setConstraints(event.detail.value); 
        setError(null);
    };

    const parseConstraints = () => {
        const lines = constraints.split("\n"); // Split input into lines
        const parsed: ConstraintItem[] = [];

        setError(null);

        for (const line of lines) {
            const regex = /(\d+)\*w(\d+)([<>]=?)(\d+)\*w(\d+)/; // Regex to capture the constraint format
            const match = line.match(regex);

            if (match && match[2] == '1' && match[5] == '2') {
                const weight1 = parseInt(match[1]); // Extract weight coefficient for w1
                const weight2 = parseInt(match[4]); // Extract weight coefficient for w2
                const operator = match[3]; // Extract operator

                const parsedConstraint: ConstraintItem = [weight1, weight2, operator];
                parsed.push(parsedConstraint);
            } else {
                const err = `
                    Invalid constraint format: "${line}".\n 
                    Expected format: "number*w1 operator number*w2" (without spaces). Example: 1*w1>=2*w2.\n
                    Every constraint should appear in a separate line.
                `;
                setError(err);
                console.error(err);
                return;
            }
        }

        setParsedConstraints(parsed); 
    };

    return (
        <SpaceBetween size="m">
            {error && <Alert type="error">{error}</Alert>}
            <Textarea
                placeholder="Enter constraints for the weight function..."
                value={constraints}
                onChange={handleConstraintsChange}
                invalid={error != null}
            />
            <Button onClick={parseConstraints}>Parse constraints</Button>
        </SpaceBetween>
    );
};